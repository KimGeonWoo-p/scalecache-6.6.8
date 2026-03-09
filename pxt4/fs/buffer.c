// SPDX-License-Identifier: GPL-2.0-only
/*
 *  linux/fs/buffer.c
 *
 *  Copyright (C) 1991, 1992, 2002  Linus Torvalds
 */

/*
 * Start bdflush() with kernel_thread not syscall - Paul Gortmaker, 12/95
 *
 * Removed a lot of unnecessary code and simplified things now that
 * the buffer cache isn't our primary cache - Andrew Tridgell 12/96
 *
 * Speed up hash, lru, and free list operations.  Use gfp() for allocating
 * hash table, use SLAB cache for buffer heads. SMP threading.  -DaveM
 *
 * Added 32k buffer block sizes - these are required older ARM systems. - RMK
 *
 * async buffer flushing, 1999 Andrea Arcangeli <andrea@suse.de>
 */

#include <linux/kernel.h>
#include <linux/sched/signal.h>
#include <linux/syscalls.h>
#include <linux/fs.h>
#include <linux/iomap.h>
#include <linux/mm.h>
#include <linux/percpu.h>
#include <linux/slab.h>
#include <linux/capability.h>
#include <linux/blkdev.h>
#include <linux/file.h>
#include <linux/quotaops.h>
#include <linux/highmem.h>
#include <linux/export.h>
#include <linux/backing-dev.h>
#include <linux/writeback.h>
#include <linux/hash.h>
#include <linux/suspend.h>
#include <linux/buffer_head.h>
#include <linux/task_io_accounting_ops.h>
#include <linux/bio.h>
#include <linux/cpu.h>
#include <linux/bitops.h>
#include <linux/mpage.h>
#include <linux/bit_spinlock.h>
#include <linux/pagevec.h>
#include <linux/sched/mm.h>
#include <trace/events/block.h>
#include <linux/fscrypt.h>
#include <linux/fsverity.h>
#include <linux/sched/isolation.h>

#include "internal.h"

static struct buffer_head *
__pxt4_find_get_block(struct block_device *bdev, sector_t block, unsigned size);

extern struct folio *
__cc_filemap_get_folio(struct address_space *mapping, pgoff_t index,
		fgf_t fgp_flags, gfp_t gfp);

/*
 * Various filesystems appear to want __find_get_block to be non-blocking.
 * But it's the page lock which protects the buffers.  To get around this,
 * we get exclusion from try_to_free_buffers with the blockdev mapping's
 * private_lock.
 *
 * Hack idea: for the blockdev mapping, private_lock contention
 * may be quite high.  This code could TryLock the page, and if that
 * succeeds, there is no need to take private_lock.
 */
static struct buffer_head *
__pxt4_find_get_block_slow(struct block_device *bdev, sector_t block)
{
	struct inode *bd_inode = bdev->bd_inode;
	struct address_space *bd_mapping = bd_inode->i_mapping;
	struct buffer_head *ret = NULL;
	pgoff_t index;
	struct buffer_head *bh;
	struct buffer_head *head;
	struct folio *folio;
	int all_mapped = 1;
	static DEFINE_RATELIMIT_STATE(last_warned, HZ, 1);
	int page_locked = 0;

	index = block >> (PAGE_SHIFT - bd_inode->i_blkbits);

	folio = __cc_filemap_get_folio(bd_mapping, index, FGP_ACCESSED, 0);
	if (IS_ERR(folio))
		goto out;

	if (!(page_locked = folio_trylock(folio)))
		spin_lock(&bd_mapping->private_lock);

	head = folio_buffers(folio);
	if (!head)
		goto out_unlock;
	bh = head;
	do {
		if (!buffer_mapped(bh))
			all_mapped = 0;
		else if (bh->b_blocknr == block) {
			ret = bh;
			get_bh(bh);
			goto out_unlock;
		}
		bh = bh->b_this_page;
	} while (bh != head);

	/* we might be here because some of the buffers on this page are
	 * not mapped.  This is due to various races between
	 * file io on the block device and getblk.  It gets dealt with
	 * elsewhere, don't buffer_error if we had some unmapped buffers
	 */
	ratelimit_set_flags(&last_warned, RATELIMIT_MSG_ON_RELEASE);
	if (all_mapped && __ratelimit(&last_warned)) {
		printk("__find_get_block_slow() failed. block=%llu, "
		       "b_blocknr=%llu, b_state=0x%08lx, b_size=%zu, "
		       "device %pg blocksize: %d\n",
		       (unsigned long long)block,
		       (unsigned long long)bh->b_blocknr,
		       bh->b_state, bh->b_size, bdev,
		       1 << bd_inode->i_blkbits);
	}
out_unlock:
	spin_unlock(&bd_mapping->private_lock);
	folio_put(folio);
out:
	return ret;
}

/*
 * fs/buffer.c contains helper functions for buffer-backed address space's
 * fsync functions.  A common requirement for buffer-based filesystems is
 * that certain data from the backing blockdev needs to be written out for
 * a successful fsync().  For example, ext2 indirect blocks need to be
 * written back and waited upon before fsync() returns.
 *
 * The functions mark_buffer_inode_dirty(), fsync_inode_buffers(),
 * inode_has_buffers() and invalidate_inode_buffers() are provided for the
 * management of a list of dependent buffers at ->i_mapping->private_list.
 *
 * Locking is a little subtle: try_to_free_buffers() will remove buffers
 * from their controlling inode's queue when they are being freed.  But
 * try_to_free_buffers() will be operating against the *blockdev* mapping
 * at the time, not against the S_ISREG file which depends on those buffers.
 * So the locking for private_list is via the private_lock in the address_space
 * which backs the buffers.  Which is different from the address_space 
 * against which the buffers are listed.  So for a particular address_space,
 * mapping->private_lock does *not* protect mapping->private_list!  In fact,
 * mapping->private_list will always be protected by the backing blockdev's
 * ->private_lock.
 *
 * Which introduces a requirement: all buffers on an address_space's
 * ->private_list must be from the same address_space: the blockdev's.
 *
 * address_spaces which do not place buffers at ->private_list via these
 * utility functions are free to use private_lock and private_list for
 * whatever they want.  The only requirement is that list_empty(private_list)
 * be true at clear_inode() time.
 *
 * FIXME: clear_inode should not call invalidate_inode_buffers().  The
 * filesystems should do that.  invalidate_inode_buffers() should just go
 * BUG_ON(!list_empty).
 *
 * FIXME: mark_buffer_dirty_inode() is a data-plane operation.  It should
 * take an address_space, not an inode.  And it should be called
 * mark_buffer_dirty_fsync() to clearly define why those buffers are being
 * queued up.
 *
 * FIXME: mark_buffer_dirty_inode() doesn't need to add the buffer to the
 * list if it is already on a list.  Because if the buffer is on a list,
 * it *must* already be on the right one.  If not, the filesystem is being
 * silly.  This will save a ton of locking.  But first we have to ensure
 * that buffers are taken *off* the old inode's list when they are freed
 * (presumably in truncate).  That requires careful auditing of all
 * filesystems (do it inside bforget()).  It could also be done by bringing
 * b_inode back.
 */

/*
 * The buffer's backing address_space's private_lock must be held
 */
static void __remove_assoc_queue(struct buffer_head *bh)
{
	list_del_init(&bh->b_assoc_buffers);
	WARN_ON(!bh->b_assoc_map);
	bh->b_assoc_map = NULL;
}

int inode_has_buffers(struct inode *inode)
{
	return !list_empty(&inode->i_data.private_list);
}

static inline void
pxt4_link_dev_buffers(struct folio *folio,
		struct buffer_head *head)
{
	struct buffer_head *bh, *tail;

	bh = head;
	do {
		tail = bh;
		bh = bh->b_this_page;
	} while (bh);
	tail->b_this_page = head;
	folio_attach_private(folio, head);
}

static sector_t pxt4_blkdev_max_block(struct block_device *bdev, unsigned int size)
{
	sector_t retval = ~((sector_t)0);
	loff_t sz = bdev_nr_bytes(bdev);

	if (sz) {
		unsigned int sizebits = blksize_bits(size);
		retval = (sz >> sizebits);
	}
	return retval;
}

/*
 * Initialise the state of a blockdev folio's buffers.
 */ 
static sector_t pxt4_folio_init_buffers(struct folio *folio,
		struct block_device *bdev, sector_t block, int size)
{
	struct buffer_head *head = folio_buffers(folio);
	struct buffer_head *bh = head;
	bool uptodate = folio_test_uptodate(folio);
	sector_t end_block = pxt4_blkdev_max_block(bdev, size);

	do {
		if (!buffer_mapped(bh)) {
			bh->b_end_io = NULL;
			bh->b_private = NULL;
			bh->b_bdev = bdev;
			bh->b_blocknr = block;
			if (uptodate)
				set_buffer_uptodate(bh);
			if (block < end_block)
				set_buffer_mapped(bh);
		}
		block++;
		bh = bh->b_this_page;
	} while (bh != head);

	/*
	 * Caller needs to validate requested block against end of device.
	 */
	return end_block;
}

extern bool pxt4_try_to_free_buffers(struct folio *page);
/*
 * Create the page-cache page that contains the requested block.
 *
 * This is used purely for blockdev mappings.
 */
static int
pxt4_grow_dev_page(struct block_device *bdev, sector_t block,
	      pgoff_t index, int size, int sizebits, gfp_t gfp)
{
	struct inode *inode = bdev->bd_inode;
	struct folio *folio;
	struct buffer_head *bh;
	sector_t end_block;
	int ret = 0;
	gfp_t gfp_mask;

	gfp_mask = mapping_gfp_constraint(inode->i_mapping, ~__GFP_FS) | gfp;

	/*
	 * XXX: __getblk_slow() can not really deal with failure and
	 * will endlessly loop on improvised global reclaim.  Prefer
	 * looping in the allocator rather than here, at least that
	 * code knows what it's doing.
	 */
	gfp_mask |= __GFP_NOFAIL;

	folio = __cc_filemap_get_folio(inode->i_mapping, index,
			FGP_LOCK | FGP_ACCESSED | FGP_CREAT, gfp_mask);

	bh = folio_buffers(folio);
	if (bh) {
		if (bh->b_size == size) {
			end_block = pxt4_folio_init_buffers(folio, bdev,
					(sector_t)index << sizebits, size);
			goto done;
		}
		if (!pxt4_try_to_free_buffers(folio))
			goto failed;
	}

	bh = folio_alloc_buffers(folio, size, true);

	/*
	 * Link the folio to the buffers and initialise them.  Take the
	 * lock to be atomic wrt __find_get_block(), which does not
	 * run under the folio lock.
	 */
	spin_lock(&inode->i_mapping->private_lock);
	pxt4_link_dev_buffers(folio, bh);
	end_block = pxt4_folio_init_buffers(folio, bdev,
			(sector_t)index << sizebits, size);
	spin_unlock(&inode->i_mapping->private_lock);
done:
	ret = (block < end_block) ? 1 : -ENXIO;
failed:
	folio_unlock(folio);
	folio_put(folio);
	return ret;
}

/*
 * Create buffers for the specified block device block's page.  If
 * that page was dirty, the buffers are set dirty also.
 */
static int
pxt4_grow_buffers(struct block_device *bdev, sector_t block, int size, gfp_t gfp)
{
	pgoff_t index;
	int sizebits;

	sizebits = PAGE_SHIFT - __ffs(size);
	index = block >> sizebits;

	/*
	 * Check for a block which wants to lie outside our maximum possible
	 * pagecache index.  (this comparison is done using sector_t types).
	 */
	if (unlikely(index != block >> sizebits)) {
		printk(KERN_ERR "%s: requested out-of-range block %llu for "
			"device %pg\n",
			__func__, (unsigned long long)block,
			bdev);
		return -EIO;
	}

	/* Create a page with the proper size buffers.. */
	return pxt4_grow_dev_page(bdev, block, index, size, sizebits, gfp);
}

static struct buffer_head *
__pxt4_getblk_slow(struct block_device *bdev, sector_t block,
	     unsigned size, gfp_t gfp)
{
	/* Size must be multiple of hard sectorsize */
	if (unlikely(size & (bdev_logical_block_size(bdev)-1) ||
			(size < 512 || size > PAGE_SIZE))) {
		printk(KERN_ERR "getblk(): invalid block size %d requested\n",
					size);
		printk(KERN_ERR "logical block size: %d\n",
					bdev_logical_block_size(bdev));

		dump_stack();
		return NULL;
	}

	for (;;) {
		struct buffer_head *bh;
		int ret;

		bh = __pxt4_find_get_block(bdev, block, size);
		if (bh)
			return bh;

		ret = pxt4_grow_buffers(bdev, block, size, gfp);
		if (ret < 0)
			return NULL;
	}
}

/*
 * Per-cpu buffer LRU implementation.  To reduce the cost of __find_get_block().
 * The bhs[] array is sorted - newest buffer is at bhs[0].  Buffers have their
 * refcount elevated by one when they're in an LRU.  A buffer can only appear
 * once in a particular CPU's LRU.  A single buffer can be present in multiple
 * CPU's LRUs at the same time.
 *
 * This is a transparent caching front-end to sb_bread(), sb_getblk() and
 * sb_find_get_block().
 *
 * The LRUs themselves only need locking against invalidate_bh_lrus.  We use
 * a local interrupt disable for that.
 */

#define BH_LRU_SIZE	16

struct bh_lru {
	struct buffer_head *bhs[BH_LRU_SIZE];
};

static DEFINE_PER_CPU(struct bh_lru, bh_lrus) = {{ NULL }};

#ifdef CONFIG_SMP
#define bh_lru_lock()	local_irq_disable()
#define bh_lru_unlock()	local_irq_enable()
#else
#define bh_lru_lock()	preempt_disable()
#define bh_lru_unlock()	preempt_enable()
#endif

static inline void check_irqs_on(void)
{
#ifdef irqs_disabled
	BUG_ON(irqs_disabled());
#endif
}

/*
 * Install a buffer_head into this cpu's LRU.  If not already in the LRU, it is
 * inserted at the front, and the buffer_head at the back if any is evicted.
 * Or, if already in the LRU it is moved to the front.
 */
static void bh_lru_install(struct buffer_head *bh)
{
	struct buffer_head *evictee = bh;
	struct bh_lru *b;
	int i;

	check_irqs_on();
	bh_lru_lock();

	/*
	 * the refcount of buffer_head in bh_lru prevents dropping the
	 * attached page(i.e., try_to_free_buffers) so it could cause
	 * failing page migration.
	 * Skip putting upcoming bh into bh_lru until migration is done.
	 */
	if (lru_cache_disabled() || cpu_is_isolated(smp_processor_id())) {
		bh_lru_unlock();
		return;
	}

	b = this_cpu_ptr(&bh_lrus);
	for (i = 0; i < BH_LRU_SIZE; i++) {
		swap(evictee, b->bhs[i]);
		if (evictee == bh) {
			bh_lru_unlock();
			return;
		}
	}

	get_bh(bh);
	bh_lru_unlock();
	brelse(evictee);
}

/*
 * Look up the bh in this cpu's LRU.  If it's there, move it to the head.
 */
static struct buffer_head *
lookup_bh_lru(struct block_device *bdev, sector_t block, unsigned size)
{
	struct buffer_head *ret = NULL;
	unsigned int i;

	check_irqs_on();
	bh_lru_lock();
	if (cpu_is_isolated(smp_processor_id())) {
		bh_lru_unlock();
		return NULL;
	}
	for (i = 0; i < BH_LRU_SIZE; i++) {
		struct buffer_head *bh = __this_cpu_read(bh_lrus.bhs[i]);

		if (bh && bh->b_blocknr == block && bh->b_bdev == bdev &&
		    bh->b_size == size) {
			if (i) {
				while (i) {
					__this_cpu_write(bh_lrus.bhs[i],
						__this_cpu_read(bh_lrus.bhs[i - 1]));
					i--;
				}
				__this_cpu_write(bh_lrus.bhs[0], bh);
			}
			get_bh(bh);
			ret = bh;
			break;
		}
	}
	bh_lru_unlock();
	return ret;
}

/*
 * Perform a pagecache lookup for the matching buffer.  If it's there, refresh
 * it in the LRU and mark it as accessed.  If it is not present then return
 * NULL
 */
static struct buffer_head *
__pxt4_find_get_block(struct block_device *bdev, sector_t block, unsigned size)
{
	struct buffer_head *bh = lookup_bh_lru(bdev, block, size);

	if (bh == NULL) {
		/* __find_get_block_slow will mark the page accessed */
		bh = __pxt4_find_get_block_slow(bdev, block);
		if (bh)
			bh_lru_install(bh);
	} else
		touch_buffer(bh);

	return bh;
}

/*
 * __getblk_gfp() will locate (and, if necessary, create) the buffer_head
 * which corresponds to the passed block_device, block and size. The
 * returned buffer has its reference count incremented.
 *
 * __getblk_gfp() will lock up the machine if grow_dev_page's
 * try_to_free_buffers() attempt is failing.  FIXME, perhaps?
 */
struct buffer_head *
__pxt4_getblk_gfp(struct block_device *bdev, sector_t block,
	     unsigned size, gfp_t gfp)
{
	struct buffer_head *bh = __pxt4_find_get_block(bdev, block, size);

	might_sleep();
	if (bh == NULL)
		bh = __pxt4_getblk_slow(bdev, block, size, gfp);
	return bh;
}

/*
 * try_to_free_buffers() checks if all the buffers on this particular folio
 * are unused, and releases them if so.
 *
 * Exclusion against try_to_free_buffers may be obtained by either
 * locking the folio or by holding its mapping's private_lock.
 *
 * If the folio is dirty but all the buffers are clean then we need to
 * be sure to mark the folio clean as well.  This is because the folio
 * may be against a block device, and a later reattachment of buffers
 * to a dirty folio will set *all* buffers dirty.  Which would corrupt
 * filesystem data on the same device.
 *
 * The same applies to regular filesystem folios: if all the buffers are
 * clean then we set the folio clean and proceed.  To do that, we require
 * total exclusion from block_dirty_folio().  That is obtained with
 * private_lock.
 *
 * try_to_free_buffers() is non-blocking.
 */
static inline int buffer_busy(struct buffer_head *bh)
{
	return atomic_read(&bh->b_count) |
		(bh->b_state & ((1 << BH_Dirty) | (1 << BH_Lock)));
}

static bool
drop_buffers(struct folio *folio, struct buffer_head **buffers_to_free)
{
	struct buffer_head *head = folio_buffers(folio);
	struct buffer_head *bh;

	bh = head;
	do {
		if (buffer_busy(bh))
			goto failed;
		bh = bh->b_this_page;
	} while (bh != head);

	do {
		struct buffer_head *next = bh->b_this_page;

		if (bh->b_assoc_map)
			__remove_assoc_queue(bh);
		bh = next;
	} while (bh != head);
	return true;
failed:
	return false;
}

bool pxt4_try_to_free_buffers(struct folio *folio)
{
	struct address_space * const mapping = folio->mapping;
	struct buffer_head *buffers_to_free = NULL;
	bool ret = 0;

	BUG_ON(!folio_test_locked(folio));
	if (folio_test_writeback(folio))
		return false;

	if (mapping == NULL) {		/* can this still happen? */
		ret = drop_buffers(folio, &buffers_to_free);
		goto out;
	}

	spin_lock(&mapping->private_lock);
	ret = drop_buffers(folio, &buffers_to_free);

	/*
	 * If the filesystem writes its buffers by hand (eg ext3)
	 * then we can have clean buffers against a dirty folio.  We
	 * clean the folio here; otherwise the VM will never notice
	 * that the filesystem did any IO at all.
	 *
	 * Also, during truncate, discard_buffer will have marked all
	 * the folio's buffers clean.  We discover that here and clean
	 * the folio also.
	 *
	 * private_lock must be held over this entire operation in order
	 * to synchronise against block_dirty_folio and prevent the
	 * dirty bit from being lost.
	 */
	if (ret)
		folio_cancel_dirty(folio);
	spin_unlock(&mapping->private_lock);
out:
	if (buffers_to_free) {
		struct buffer_head *bh = buffers_to_free;

		do {
			struct buffer_head *next = bh->b_this_page;
			free_buffer_head(bh);
			bh = next;
		} while (bh != buffers_to_free);
	}
	return ret;
}
EXPORT_SYMBOL(pxt4_try_to_free_buffers);

