// SPDX-License-Identifier: GPL-2.0-only
/*
 * fs/fs-writeback.c
 *
 * Copyright (C) 2002, Linus Torvalds.
 *
 * Contains all the functions related to writing back and waiting
 * upon dirty inodes against superblocks, and writing back dirty
 * pages against inodes.  ie: data writeback.  Writeout of the
 * inode itself is not handled here.
 *
 * 10Apr2002	Andrew Morton
 *		Split out of fs/inode.c
 *		Additions for address_space-based writeback
 */

#include <linux/kernel.h>
#include <linux/export.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/pagemap.h>
#include <linux/kthread.h>
#include <linux/writeback.h>
#include <linux/blkdev.h>
#include <linux/backing-dev.h>
#include <linux/tracepoint.h>
#include <linux/device.h>
#include <linux/memcontrol.h>
#include "internal.h"

/*
 * 4MB minimal write chunk size
 */
#define MIN_WRITEBACK_PAGES	(4096UL >> (PAGE_SHIFT - 10))

 struct wb_writeback_work {
	long nr_pages;
	struct super_block *sb;
	enum writeback_sync_modes sync_mode;
	unsigned int tagged_writepages:1;
	unsigned int for_kupdate:1;
	unsigned int range_cyclic:1;
	unsigned int for_background:1;
	unsigned int for_sync:1;	/* sync(2) WB_SYNC_ALL writeback */
	unsigned int auto_free:1;       /* free on completion */
	enum wb_reason reason;	  /* why was writeback initiated? */

	struct list_head list;	  /* pending work list */
	struct wb_completion *done;     /* set if the caller waits */
};

/*
 * If an inode is constantly having its pages dirtied, but then the
 * updates stop dirtytime_expire_interval seconds in the past, it's
 * possible for the worst case time between when an inode has its
 * timestamps updated and when they finally get written out to be two
 * dirtytime_expire_intervals.  We set the default to 12 hours (in
 * seconds), which means most of the time inodes will have their
 * timestamps written to disk after 12 hours, but in the worst case a
 * few inodes might not their timestamps updated for 24 hours.
 */
unsigned int dirtytime_expire_interval = 12 * 60 * 60;

inline struct inode *wb_inode(struct list_head *head)
{
	return list_entry(head, struct inode, i_io_list);
}
// EXPORT_SYMBOL(wb_inode);

static bool pxt4_wb_io_lists_populated(struct bdi_writeback *wb)
{
	if (wb_has_dirty_io(wb)) {
		return false;
	} else {
		set_bit(WB_has_dirty_io, &wb->state);
		WARN_ON_ONCE(!wb->avg_write_bandwidth);
		atomic_long_add(wb->avg_write_bandwidth,
				&wb->bdi->tot_write_bandwidth);
		return true;
	}
}

static void pxt4_wb_io_lists_depopulated(struct bdi_writeback *wb)
{
	if (wb_has_dirty_io(wb) && list_empty(&wb->b_dirty) &&
	    list_empty(&wb->b_io) && list_empty(&wb->b_more_io)) {
		clear_bit(WB_has_dirty_io, &wb->state);
		WARN_ON_ONCE(atomic_long_sub_return(wb->avg_write_bandwidth,
					&wb->bdi->tot_write_bandwidth) < 0);
	}
}

static struct bdi_writeback *pxt4_inode_to_wb_and_lock_list(struct inode *inode); 
/*
 * Remove the inode from the writeback list it is on.
 */
static void pxt4_inode_io_list_del(struct inode *inode)
{
	struct bdi_writeback *wb;

	wb = pxt4_inode_to_wb_and_lock_list(inode);
	spin_lock(&inode->i_lock);

	inode->i_state &= ~I_SYNC_QUEUED;
	list_del_init(&inode->i_io_list);
	pxt4_wb_io_lists_depopulated(wb);

	spin_unlock(&inode->i_lock);
	spin_unlock(&wb->list_lock);
}

static struct bdi_writeback *
pxt4_locked_inode_to_wb_and_lock_list(struct inode *inode)
	__releases(&inode->i_lock)
	__acquires(&wb->list_lock)
{
	while (true) {
		struct bdi_writeback *wb = inode_to_wb(inode);
		wb_get(wb);
		spin_unlock(&inode->i_lock);
		spin_lock(&wb->list_lock);

		if (likely(wb == inode->i_wb)) {
			wb_put(wb);
			return wb;
		}

		spin_unlock(&wb->list_lock);
		wb_put(wb);
		cpu_relax();
		spin_lock(&inode->i_lock);
	}
}

static struct bdi_writeback *pxt4_inode_to_wb_and_lock_list(struct inode *inode)
	__acquires(&wb->list_lock)
{
	spin_lock(&inode->i_lock);
	return pxt4_locked_inode_to_wb_and_lock_list(inode);
}

/**
 * inode_io_list_move_locked - move an inode onto a bdi_writeback IO list
 * @inode: inode to be moved
 * @wb: target bdi_writeback
 * @head: one of @wb->b_{dirty|io|more_io|dirty_time}
 *
 * Move @inode->i_io_list to @list of @wb and set %WB_has_dirty_io.
 * Returns %true if @inode is the first occupant of the !dirty_time IO
 * lists; otherwise, %false.
 */
static bool pxt4_inode_io_list_move_locked(struct inode *inode,
                                      struct bdi_writeback *wb,
                                      struct list_head *head)
{
        assert_spin_locked(&wb->list_lock);
        assert_spin_locked(&inode->i_lock);
        WARN_ON_ONCE(inode->i_state & I_FREEING);

        list_move(&inode->i_io_list, head);

        /* dirty_time doesn't count as dirty_io until expiration */
        if (head != &wb->b_dirty_time)
                return pxt4_wb_io_lists_populated(wb);

        pxt4_wb_io_lists_depopulated(wb);
        return false;
}

/*
 * Redirty an inode: set its when-it-was dirtied timestamp and move it to the
 * furthest end of its superblock's dirty-inode list.
 *
 * Before stamping the inode's ->dirtied_when, we check to see whether it is
 * already the most-recently-dirtied inode on the b_dirty list.  If that is
 * the case then the inode must have been redirtied while it was being written
 * out and we don't reset its dirtied_when.
 */
static void pxt4_redirty_tail_locked(struct inode *inode, struct bdi_writeback *wb)
{
	assert_spin_locked(&inode->i_lock);

	inode->i_state &= ~I_SYNC_QUEUED;
	/*
	 * When the inode is being freed just don't bother with dirty list
	 * tracking. Flush worker will ignore this inode anyway and it will
	 * trigger assertions in inode_io_list_move_locked().
	 */
	if (inode->i_state & I_FREEING) {
		list_del_init(&inode->i_io_list);
		pxt4_wb_io_lists_depopulated(wb);
		return;
	}
	if (!list_empty(&wb->b_dirty)) {
		struct inode *tail;

		tail = wb_inode(wb->b_dirty.next);
		if (time_before(inode->dirtied_when, tail->dirtied_when))
			inode->dirtied_when = jiffies;
	}
	pxt4_inode_io_list_move_locked(inode, wb, &wb->b_dirty);
}

static void pxt4_redirty_tail(struct inode *inode, struct bdi_writeback *wb)
{
	spin_lock(&inode->i_lock);
	pxt4_redirty_tail_locked(inode, wb);
	spin_unlock(&inode->i_lock);
}

/*
 * requeue inode for re-scanning after bdi->b_io list is exhausted.
 */
static void pxt4_requeue_io(struct inode *inode, struct bdi_writeback *wb)
{
	pxt4_inode_io_list_move_locked(inode, wb, &wb->b_more_io);
}

static void pxt4_inode_sync_complete(struct inode *inode)
{
	inode->i_state &= ~I_SYNC;
	/* If inode is clean an unused, put it into LRU now... */
	inode_add_lru(inode);
	/* Waiters must see I_SYNC cleared before being woken up */
	smp_mb();
	wake_up_bit(&inode->i_state, __I_SYNC);
}

static bool inode_dirtied_after(struct inode *inode, unsigned long t)
{
	bool ret = time_after(inode->dirtied_when, t);
#ifndef CONFIG_64BIT
	/*
	 * For inodes being constantly redirtied, dirtied_when can get stuck.
	 * It _appears_ to be in the future, but is actually in distant past.
	 * This test is necessary to prevent such wrapped-around relative times
	 * from permanently stopping the whole bdi writeback.
	 */
	ret = ret && time_before_eq(inode->dirtied_when, jiffies);
#endif
	return ret;
}

/*
 * Move expired (dirtied before dirtied_before) dirty inodes from
 * @delaying_queue to @dispatch_queue.
 */
static int move_expired_inodes(struct list_head *delaying_queue,
			       struct list_head *dispatch_queue,
			       unsigned long dirtied_before)
{
	LIST_HEAD(tmp);
	struct list_head *pos, *node;
	struct super_block *sb = NULL;
	struct inode *inode;
	int do_sb_sort = 0;
	int moved = 0;

	while (!list_empty(delaying_queue)) {
		inode = wb_inode(delaying_queue->prev);
		if (inode_dirtied_after(inode, dirtied_before))
			break;
		spin_lock(&inode->i_lock);
		list_move(&inode->i_io_list, &tmp);
		moved++;
		inode->i_state |= I_SYNC_QUEUED;
		spin_unlock(&inode->i_lock);
		if (sb_is_blkdev_sb(inode->i_sb))
			continue;
		if (sb && sb != inode->i_sb)
			do_sb_sort = 1;
		sb = inode->i_sb;
	}

	/* just one sb in list, splice to dispatch_queue and we're done */
	if (!do_sb_sort) {
		list_splice(&tmp, dispatch_queue);
		goto out;
	}

	/*
	 * Although inode's i_io_list is moved from 'tmp' to 'dispatch_queue',
	 * we don't take inode->i_lock here because it is just a pointless overhead.
	 * Inode is already marked as I_SYNC_QUEUED so writeback list handling is
	 * fully under our control.
	 */
	while (!list_empty(&tmp)) {
		sb = wb_inode(tmp.prev)->i_sb;
		list_for_each_prev_safe(pos, node, &tmp) {
			inode = wb_inode(pos);
			if (inode->i_sb == sb)
				list_move(&inode->i_io_list, dispatch_queue);
		}
	}
out:
	return moved;
}

/*
 * Queue all expired dirty inodes for io, eldest first.
 * Before
 *	 newly dirtied     b_dirty    b_io    b_more_io
 *	 =============>    gf	 edc     BA
 * After
 *	 newly dirtied     b_dirty    b_io    b_more_io
 *	 =============>    g	  fBAedc
 *					   |
 *					   +--> dequeue for IO
 */
// static void queue_io(struct bdi_writeback *wb, struct wb_writeback_work *work,
//		     unsigned long dirtied_before)

void queue_io(struct bdi_writeback *wb, struct wb_writeback_work *work,
		     unsigned long dirtied_before)
{
	int moved;
	unsigned long time_expire_jif = dirtied_before;

	assert_spin_locked(&wb->list_lock);
	list_splice_init(&wb->b_more_io, &wb->b_io);
	moved = move_expired_inodes(&wb->b_dirty, &wb->b_io, dirtied_before);
	if (!work->for_sync)
		time_expire_jif = jiffies - dirtytime_expire_interval * HZ;
	moved += move_expired_inodes(&wb->b_dirty_time, &wb->b_io,
				     time_expire_jif);
	if (moved)
		pxt4_wb_io_lists_populated(wb);
	//trace_writeback_queue_io(wb, work, dirtied_before, moved);
}
//EXPORT_SYMBOL(queue_io);

/*
 * requeue inode for re-scanning after bdi->b_io list is exhausted.
 */
static void requeue_io(struct inode *inode, struct bdi_writeback *wb)
{
        pxt4_inode_io_list_move_locked(inode, wb, &wb->b_more_io);
}

static void inode_sync_complete(struct inode *inode)
{
        inode->i_state &= ~I_SYNC;
        /* If inode is clean an unused, put it into LRU now... */
        inode_add_lru(inode);
        /* Waiters must see I_SYNC cleared before being woken up */
        smp_mb();
        wake_up_bit(&inode->i_state, __I_SYNC);
}

static int pxt4_write_inode(struct inode *inode, struct writeback_control *wbc)
{
	int ret;

	if (inode->i_sb->s_op->write_inode && !is_bad_inode(inode)) {
		//trace_writeback_write_inode_start(inode, wbc);
		ret = inode->i_sb->s_op->write_inode(inode, wbc);
		//trace_writeback_write_inode(inode, wbc);
		return ret;
	}
	return 0;
}

/*
 * Sleep until I_SYNC is cleared. This function must be called with i_lock
 * held and drops it. It is aimed for callers not holding any inode reference
 * so once i_lock is dropped, inode can go away.
 */
static void pxt4_inode_sleep_on_writeback(struct inode *inode)
	__releases(inode->i_lock)
{
	DEFINE_WAIT(wait);
	wait_queue_head_t *wqh = bit_waitqueue(&inode->i_state, __I_SYNC);
	int sleep;

	prepare_to_wait(wqh, &wait, TASK_UNINTERRUPTIBLE);
	sleep = inode->i_state & I_SYNC;
	spin_unlock(&inode->i_lock);
	if (sleep)
		schedule();
	finish_wait(wqh, &wait);
}


static void inode_cgwb_move_to_attached(struct inode *inode,
					struct bdi_writeback *wb)
{
	assert_spin_locked(&wb->list_lock);
	assert_spin_locked(&inode->i_lock);
	WARN_ON_ONCE(inode->i_state & I_FREEING);

	inode->i_state &= ~I_SYNC_QUEUED;
	if (wb != &wb->bdi->wb)
		list_move(&inode->i_io_list, &wb->b_attached);
	else
		list_del_init(&inode->i_io_list);
	pxt4_wb_io_lists_depopulated(wb);
}

/*
 * Write out an inode and its dirty pages (or some of its dirty pages, depending
 * on @wbc->nr_to_write), and clear the relevant dirty flags from i_state.
 *
 * This doesn't remove the inode from the writeback list it is on, except
 * potentially to move it from b_dirty_time to b_dirty due to timestamp
 * expiration.  The caller is otherwise responsible for writeback list handling.
 *
 * The caller is also responsible for setting the I_SYNC flag beforehand and
 * calling inode_sync_complete() to clear it afterwards.
 */
static int
__writeback_single_inode(struct inode *inode, struct writeback_control *wbc)
{
        struct address_space *mapping = inode->i_mapping;
        long nr_to_write = wbc->nr_to_write;
        unsigned dirty;
        int ret;

        WARN_ON(!(inode->i_state & I_SYNC));

//        trace_writeback_single_inode_start(inode, wbc, nr_to_write);

        ret = do_writepages(mapping, wbc);

        /*
         * Make sure to wait on the data before writing out the metadata.
         * This is important for filesystems that modify metadata on data
         * I/O completion. We don't do it for sync(2) writeback because it has a
         * separate, external IO completion path and ->sync_fs for guaranteeing
         * inode metadata is written back correctly.
         */
        if (wbc->sync_mode == WB_SYNC_ALL && !wbc->for_sync) {
                int err = filemap_fdatawait(mapping);
                if (ret == 0)
                        ret = err;
        }

        /*
         * If the inode has dirty timestamps and we need to write them, call
         * mark_inode_dirty_sync() to notify the filesystem about it and to
         * change I_DIRTY_TIME into I_DIRTY_SYNC.
         */
        if ((inode->i_state & I_DIRTY_TIME) &&
            (wbc->sync_mode == WB_SYNC_ALL ||
             time_after(jiffies, inode->dirtied_time_when +
                        dirtytime_expire_interval * HZ))) {
  //              trace_writeback_lazytime(inode);
                mark_inode_dirty_sync(inode);
        }

        /*
         * Get and clear the dirty flags from i_state.  This needs to be done
         * after calling writepages because some filesystems may redirty the
         * inode during writepages due to delalloc.  It also needs to be done
         * after handling timestamp expiration, as that may dirty the inode too.
         */
        spin_lock(&inode->i_lock);
        dirty = inode->i_state & I_DIRTY;
        inode->i_state &= ~dirty;

        /*
         * Paired with smp_mb() in __mark_inode_dirty().  This allows
         * __mark_inode_dirty() to test i_state without grabbing i_lock -
         * either they see the I_DIRTY bits cleared or we see the dirtied
         * inode.
         *
         * I_DIRTY_PAGES is always cleared together above even if @mapping
         * still has dirty pages.  The flag is reinstated after smp_mb() if
         * necessary.  This guarantees that either __mark_inode_dirty()
         * sees clear I_DIRTY_PAGES or we see PAGECACHE_TAG_DIRTY.
         */
        smp_mb();

        if (mapping_tagged(mapping, PAGECACHE_TAG_DIRTY))
                inode->i_state |= I_DIRTY_PAGES;
        else if (unlikely(inode->i_state & I_PINNING_FSCACHE_WB)) {
                if (!(inode->i_state & I_DIRTY_PAGES)) {
                        inode->i_state &= ~I_PINNING_FSCACHE_WB;
                        wbc->unpinned_fscache_wb = true;
                        dirty |= I_PINNING_FSCACHE_WB; /* Cause write_inode */
                }
        }

        spin_unlock(&inode->i_lock);

        /* Don't write the inode if only I_DIRTY_PAGES was set */
        if (dirty & ~I_DIRTY_PAGES) {
                int err = pxt4_write_inode(inode, wbc);
                if (ret == 0)
                        ret = err;
        }
        wbc->unpinned_fscache_wb = false;
//        trace_writeback_single_inode(inode, wbc, nr_to_write);
        return ret;
}

/*
 * Find proper writeback list for the inode depending on its current state and
 * possibly also change of its state while we were doing writeback.  Here we
 * handle things such as livelock prevention or fairness of writeback among
 * inodes. This function can be called only by flusher thread - noone else
 * processes all inodes in writeback lists and requeueing inodes behind flusher
 * thread's back can have unexpected consequences.
 */
static void pxt4_requeue_inode(struct inode *inode, struct bdi_writeback *wb,
			  struct writeback_control *wbc)
{
	if (inode->i_state & I_FREEING)
		return;

	/*
	 * Sync livelock prevention. Each inode is tagged and synced in one
	 * shot. If still dirty, it will be redirty_tail()'ed below.  Update
	 * the dirty time to prevent enqueue and sync it again.
	 */
	if ((inode->i_state & I_DIRTY) &&
	    (wbc->sync_mode == WB_SYNC_ALL || wbc->tagged_writepages))
		inode->dirtied_when = jiffies;

	if (wbc->pages_skipped) {
		/*
		 * Writeback is not making progress due to locked buffers.
		 * Skip this inode for now. Although having skipped pages
		 * is odd for clean inodes, it can happen for some
		 * filesystems so handle that gracefully.
		 */
		if (inode->i_state & I_DIRTY_ALL)
			pxt4_redirty_tail_locked(inode, wb);
		else
			inode_cgwb_move_to_attached(inode, wb);
		return;
	}

	if (mapping_tagged(inode->i_mapping, PAGECACHE_TAG_DIRTY)) {
		/*
		 * We didn't write back all the pages.  nfs_writepages()
		 * sometimes bales out without doing anything.
		 */
		if (wbc->nr_to_write <= 0) {
			/* Slice used up. Queue for next turn. */
			pxt4_requeue_io(inode, wb);
		} else {
			/*
			 * Writeback blocked by something other than
			 * congestion. Delay the inode for some time to
			 * avoid spinning on the CPU (100% iowait)
			 * retrying writeback of the dirty page/inode
			 * that cannot be performed immediately.
			 */
			pxt4_redirty_tail_locked(inode, wb);
		}
	} else if (inode->i_state & I_DIRTY) {
		/*
		 * Filesystems can dirty the inode during writeback operations,
		 * such as delayed allocation during submission or metadata
		 * updates after data IO completion.
		 */
		pxt4_redirty_tail_locked(inode, wb);
	} else if (inode->i_state & I_DIRTY_TIME) {
		inode->dirtied_when = jiffies;
		pxt4_inode_io_list_move_locked(inode, wb, &wb->b_dirty_time);
		inode->i_state &= ~I_SYNC_QUEUED;
	} else {
		/* The inode is clean. Remove from writeback lists. */
		inode_cgwb_move_to_attached(inode, wb);
	}
}

extern void wb_wakeup(struct bdi_writeback *wb);

/*
 * Return the next wb_writeback_work struct that hasn't been processed yet.
 */
static struct wb_writeback_work *pxt4_get_next_work_item(struct bdi_writeback *wb)
{
        struct wb_writeback_work *work = NULL;

        spin_lock_irq(&wb->work_lock);
        if (!list_empty(&wb->work_list)) {
                work = list_entry(wb->work_list.next,
                                  struct wb_writeback_work, list);
                list_del_init(&work->list);
        }
        spin_unlock_irq(&wb->work_lock);
        return work;
}

static long writeback_chunk_size(struct bdi_writeback *wb,
                                 struct wb_writeback_work *work)
{
        long pages;

        /*
         * WB_SYNC_ALL mode does livelock avoidance by syncing dirty
         * inodes/pages in one big loop. Setting wbc.nr_to_write=LONG_MAX
         * here avoids calling into writeback_inodes_wb() more than once.
         *
         * The intended call sequence for WB_SYNC_ALL writeback is:
         *
         *      wb_writeback()
         *          writeback_sb_inodes()       <== called only once
         *              write_cache_pages()     <== called once for each inode
         *                   (quickly) tag currently dirty pages
         *                   (maybe slowly) sync all tagged pages
         */
        if (work->sync_mode == WB_SYNC_ALL || work->tagged_writepages)
                pages = LONG_MAX;
        else {
                pages = min(wb->avg_write_bandwidth / 2,
                            global_wb_domain.dirty_limit / DIRTY_SCOPE);
                pages = min(pages, work->nr_pages);
                pages = round_down(pages + MIN_WRITEBACK_PAGES,
                                   MIN_WRITEBACK_PAGES);
        }

        return pages;
}

static long writeback_sb_inodes(struct super_block *sb,
                                struct bdi_writeback *wb,
                                struct wb_writeback_work *work)
{
        struct writeback_control wbc = {
                .sync_mode              = work->sync_mode,
                .tagged_writepages      = work->tagged_writepages,
                .for_kupdate            = work->for_kupdate,
                .for_background         = work->for_background,
                .for_sync               = work->for_sync,
                .range_cyclic           = work->range_cyclic,
                .range_start            = 0,
                .range_end              = LLONG_MAX,
        };
        unsigned long start_time = jiffies;
        long write_chunk;
        long total_wrote = 0;  /* count both pages and inodes */

        while (!list_empty(&wb->b_io)) {
                struct inode *inode = wb_inode(wb->b_io.prev);
                struct bdi_writeback *tmp_wb;
                long wrote;

                if (inode->i_sb != sb) {
                        if (work->sb) {
                                /*
                                 * We only want to write back data for this
                                 * superblock, move all inodes not belonging
                                 * to it back onto the dirty list.
                                 */
                                pxt4_redirty_tail(inode, wb);
                                continue;
                        }

                        /*
                         * The inode belongs to a different superblock.
                         * Bounce back to the caller to unpin this and
                         * pin the next superblock.
                         */
                        break;
                }

                /*
                 * Don't bother with new inodes or inodes being freed, first
                 * kind does not need periodic writeout yet, and for the latter
                 * kind writeout is handled by the freer.
                 */
                spin_lock(&inode->i_lock);
                if (inode->i_state & (I_NEW | I_FREEING | I_WILL_FREE)) {
                        pxt4_redirty_tail_locked(inode, wb);
                        spin_unlock(&inode->i_lock);
                        continue;
                }
                if ((inode->i_state & I_SYNC) && wbc.sync_mode != WB_SYNC_ALL) {
                        /*
                         * If this inode is locked for writeback and we are not
                         * doing writeback-for-data-integrity, move it to
                         * b_more_io so that writeback can proceed with the
                         * other inodes on s_io.
                         *
                         * We'll have another go at writing back this inode
                         * when we completed a full scan of b_io.
                         */
                        requeue_io(inode, wb);
                        spin_unlock(&inode->i_lock);
                        // trace_writeback_sb_inodes_requeue(inode);
                        continue;
                }
                spin_unlock(&wb->list_lock);

                /*
                 * We already requeued the inode if it had I_SYNC set and we
                 * are doing WB_SYNC_NONE writeback. So this catches only the
                 * WB_SYNC_ALL case.
                 */
                if (inode->i_state & I_SYNC) {
                        /* Wait for I_SYNC. This function drops i_lock... */
                        pxt4_inode_sleep_on_writeback(inode);
                        /* Inode may be gone, start again */
                        spin_lock(&wb->list_lock);
                        continue;
                }
                inode->i_state |= I_SYNC;
                wbc_attach_and_unlock_inode(&wbc, inode);

                write_chunk = writeback_chunk_size(wb, work);
                wbc.nr_to_write = write_chunk;
                wbc.pages_skipped = 0;

                /*
                 * We use I_SYNC to pin the inode in memory. While it is set
                 * evict_inode() will wait so the inode cannot be freed.
                 */
                __writeback_single_inode(inode, &wbc);

                wbc_detach_inode(&wbc);
                work->nr_pages -= write_chunk - wbc.nr_to_write;
                wrote = write_chunk - wbc.nr_to_write - wbc.pages_skipped;
                wrote = wrote < 0 ? 0 : wrote;
                total_wrote += wrote;

                if (need_resched()) {
                        /*
                         * We're trying to balance between building up a nice
                         * long list of IOs to improve our merge rate, and
                         * getting those IOs out quickly for anyone throttling
                         * in balance_dirty_pages().  cond_resched() doesn't
                         * unplug, so get our IOs out the door before we
                         * give up the CPU.
                         */
                        blk_flush_plug(current->plug, false);
                        cond_resched();
                }

                /*
                 * Requeue @inode if still dirty.  Be careful as @inode may
                 * have been switched to another wb in the meantime.
                 */
                tmp_wb = pxt4_inode_to_wb_and_lock_list(inode);
                spin_lock(&inode->i_lock);
                if (!(inode->i_state & I_DIRTY_ALL))
                        total_wrote++;
                pxt4_requeue_inode(inode, tmp_wb, &wbc);
                pxt4_inode_sync_complete(inode);
                spin_unlock(&inode->i_lock);

                if (unlikely(tmp_wb != wb)) {
                        spin_unlock(&tmp_wb->list_lock);
                        spin_lock(&wb->list_lock);
                }

                /*
                 * bail out to wb_writeback() often enough to check
                 * background threshold and other termination conditions.
                 */
                if (total_wrote) {
                        if (time_is_before_jiffies(start_time + HZ / 10UL))
                                break;
                        if (work->nr_pages <= 0)
                                break;
                }
        }
        return total_wrote;
}

static long __writeback_inodes_wb(struct bdi_writeback *wb,
                                  struct wb_writeback_work *work)
{
        unsigned long start_time = jiffies;
        long wrote = 0;

        while (!list_empty(&wb->b_io)) {
                struct inode *inode = wb_inode(wb->b_io.prev);
                struct super_block *sb = inode->i_sb;

                if (!super_trylock_shared(sb)) {
                        /*
                         * super_trylock_shared() may fail consistently due to
                         * s_umount being grabbed by someone else. Don't use
                         * requeue_io() to avoid busy retrying the inode/sb.
                         */
                        pxt4_redirty_tail(inode, wb);
                        continue;
                }
                wrote += writeback_sb_inodes(sb, wb, work);
                up_read(&sb->s_umount);

                /* refer to the same tests at the end of writeback_sb_inodes */
                if (wrote) {
                        if (time_is_before_jiffies(start_time + HZ / 10UL))
                                break;
                        if (work->nr_pages <= 0)
                                break;
                }
        }
        /* Leave any unwritten inodes on b_io */
        return wrote;
}

static long writeback_inodes_wb(struct bdi_writeback *wb, long nr_pages,
                                enum wb_reason reason)
{
        struct wb_writeback_work work = {
                .nr_pages       = nr_pages,
                .sync_mode      = WB_SYNC_NONE,
                .range_cyclic   = 1,
                .reason         = reason,
        };
        struct blk_plug plug;

        blk_start_plug(&plug);
        spin_lock(&wb->list_lock);
        if (list_empty(&wb->b_io))
                queue_io(wb, &work, jiffies);
        __writeback_inodes_wb(wb, &work);
        spin_unlock(&wb->list_lock);
        blk_finish_plug(&plug);

        return nr_pages - work.nr_pages;
}


static long wb_writeback(struct bdi_writeback *wb,
                         struct wb_writeback_work *work)
{
        long nr_pages = work->nr_pages;
        unsigned long dirtied_before = jiffies;
        struct inode *inode;
        long progress;
        struct blk_plug plug;

        blk_start_plug(&plug);
        for (;;) {
                /*
                 * Stop writeback when nr_pages has been consumed
                 */
                if (work->nr_pages <= 0)
                        break;

                /*
                 * Background writeout and kupdate-style writeback may
                 * run forever. Stop them if there is other work to do
                 * so that e.g. sync can proceed. They'll be restarted
                 * after the other works are all done.
                 */
                if ((work->for_background || work->for_kupdate) &&
                    !list_empty(&wb->work_list))
                        break;

                /*
                 * For background writeout, stop when we are below the
                 * background dirty threshold
                 */
                if (work->for_background && !wb_over_bg_thresh(wb))
                        break;


                spin_lock(&wb->list_lock);

                /*
                 * Kupdate and background works are special and we want to
                 * include all inodes that need writing. Livelock avoidance is
                 * handled by these works yielding to any other work so we are
                 * safe.
                 */
                if (work->for_kupdate) {
                        dirtied_before = jiffies -
                                msecs_to_jiffies(dirty_expire_interval * 10);
                } else if (work->for_background)
                        dirtied_before = jiffies;

//                trace_writeback_start(wb, work);
                if (list_empty(&wb->b_io))
                        queue_io(wb, work, dirtied_before);
                if (work->sb)
                        progress = writeback_sb_inodes(work->sb, wb, work);
                else
                        progress = __writeback_inodes_wb(wb, work);
//                trace_writeback_written(wb, work);

                /*
                 * Did we write something? Try for more
                 *
                 * Dirty inodes are moved to b_io for writeback in batches.
                 * The completion of the current batch does not necessarily
                 * mean the overall work is done. So we keep looping as long
                 * as made some progress on cleaning pages or inodes.
                 */
                if (progress) {
                        spin_unlock(&wb->list_lock);
                        continue;
                }

                /*
                 * No more inodes for IO, bail
                 */
                if (list_empty(&wb->b_more_io)) {
                        spin_unlock(&wb->list_lock);
                        break;
                }

                /*
                 * Nothing written. Wait for some inode to
                 * become available for writeback. Otherwise
                 * we'll just busyloop.
                 */
//                trace_writeback_wait(wb, work);
                inode = wb_inode(wb->b_more_io.prev);
                spin_lock(&inode->i_lock);
                spin_unlock(&wb->list_lock);
                /* This function drops i_lock... */
                pxt4_inode_sleep_on_writeback(inode);
        }
        blk_finish_plug(&plug);

        return nr_pages - work->nr_pages;
}

void finish_writeback_work(struct bdi_writeback *wb,
                                  struct wb_writeback_work *work)
{
        struct wb_completion *done = work->done;

        if (work->auto_free)
                kfree(work);
        if (done) {
                wait_queue_head_t *waitq = done->waitq;

                /* @done can't be accessed after the following dec */
                if (atomic_dec_and_test(&done->cnt))
                        wake_up_all(waitq);
        }
}
//EXPORT_SYMBOL(wb_wait_for_completion);

/*
 * Add in the number of potentially dirty inodes, because each inode
 * write can dirty pagecache in the underlying blockdev.
 */
//static unsigned long get_nr_dirty_pages(void)
unsigned long get_nr_dirty_pages(void)
{
        return global_node_page_state(NR_FILE_DIRTY) +
                get_nr_dirty_inodes();
}
//EXPORT_SYMBOL(get_nr_dirty_pages);

/**
 * wb_split_bdi_pages - split nr_pages to write according to bandwidth
 * @wb: target bdi_writeback to split @nr_pages to
 * @nr_pages: number of pages to write for the whole bdi
 *
 * Split @wb's portion of @nr_pages according to @wb's write bandwidth in
 * relation to the total write bandwidth of all wb's w/ dirty inodes on
 * @wb->bdi.
 */
// static long wb_split_bdi_pages(struct bdi_writeback *wb, long nr_pages)
long wb_split_bdi_pages(struct bdi_writeback *wb, long nr_pages)
{
        unsigned long this_bw = wb->avg_write_bandwidth;
        unsigned long tot_bw = atomic_long_read(&wb->bdi->tot_write_bandwidth);

        if (nr_pages == LONG_MAX)
                return LONG_MAX;

        /*
         * This may be called on clean wb's and proportional distribution
         * may not make sense, just use the original @nr_pages in those
         * cases.  In general, we wanna err on the side of writing more.
         */
        if (!tot_bw || this_bw >= tot_bw)
                return nr_pages;
        else
                return DIV_ROUND_UP_ULL((u64)nr_pages * this_bw, tot_bw);
}
//EXPORT_SYMBOL(wb_split_bdi_pages);

// static long wb_check_start_all(struct bdi_writeback *wb)
long wb_check_start_all(struct bdi_writeback *wb)
{
        long nr_pages;

        if (!test_bit(WB_start_all, &wb->state))
                return 0;

        nr_pages = get_nr_dirty_pages();
        if (nr_pages) {
                struct wb_writeback_work work = {
                        .nr_pages       = wb_split_bdi_pages(wb, nr_pages),
                        .sync_mode      = WB_SYNC_NONE,
                        .range_cyclic   = 1,
                        .reason         = wb->start_all_reason,
                };

                nr_pages = wb_writeback(wb, &work);
        }

        clear_bit(WB_start_all, &wb->state);
        return nr_pages;
}
//EXPORT_SYMBOL(wb_check_start_all);

static long wb_check_background_flush(struct bdi_writeback *wb)
{
        if (wb_over_bg_thresh(wb)) {

                struct wb_writeback_work work = {
                        .nr_pages       = LONG_MAX,
                        .sync_mode      = WB_SYNC_NONE,
                        .for_background = 1,
                        .range_cyclic   = 1,
                        .reason         = WB_REASON_BACKGROUND,
                };

                return wb_writeback(wb, &work);
        }

        return 0;
}
long wb_check_old_data_flush(struct bdi_writeback *wb)
{
        unsigned long expired;
        long nr_pages;

        /*
         * When set to zero, disable periodic writeback
         */
        if (!dirty_writeback_interval)
                return 0;

        expired = wb->last_old_flush +
                        msecs_to_jiffies(dirty_writeback_interval * 10);
        if (time_before(jiffies, expired))
                return 0;

        wb->last_old_flush = jiffies;
        nr_pages = get_nr_dirty_pages();

        if (nr_pages) {
                struct wb_writeback_work work = {
                        .nr_pages       = nr_pages,
                        .sync_mode      = WB_SYNC_NONE,
                        .for_kupdate    = 1,
                        .range_cyclic   = 1,
                        .reason         = WB_REASON_PERIODIC,
                };

                return wb_writeback(wb, &work);
        }

        return 0;
}
//EXPORT_SYMBOL(wb_check_old_data_flush);
/*
 * Retrieve work items and do the writeback they describe
 */
// static long wb_do_writeback(struct bdi_writeback *wb)
long pxt4_wb_do_writeback(struct bdi_writeback *wb)
{
        struct wb_writeback_work *work;
        long wrote = 0;

        set_bit(WB_writeback_running, &wb->state);
        while ((work = pxt4_get_next_work_item(wb)) != NULL) {
//                trace_writeback_exec(wb, work);
                wrote += wb_writeback(wb, work);
                finish_writeback_work(wb, work);
        }

        /*
         * Check for a flush-everything request
         */
        wrote += wb_check_start_all(wb);

        /*
         * Check for periodic writeback, kupdated() style
         */
        wrote += wb_check_old_data_flush(wb);
        wrote += wb_check_background_flush(wb);
        clear_bit(WB_writeback_running, &wb->state);

        return wrote;
}
//EXPORT_SYMBOL(wb_do_writeback);

/*
 * Sleep until I_SYNC is cleared. This function must be called with i_lock
 * held and drops it. It is aimed for callers not holding any inode reference
 * so once i_lock is dropped, inode can go away.
 */
static void inode_sleep_on_writeback(struct inode *inode)
        __releases(inode->i_lock)
{
        DEFINE_WAIT(wait);
        wait_queue_head_t *wqh = bit_waitqueue(&inode->i_state, __I_SYNC);
        int sleep;

        prepare_to_wait(wqh, &wait, TASK_UNINTERRUPTIBLE);
        sleep = inode->i_state & I_SYNC;
        spin_unlock(&inode->i_lock);
        if (sleep)
                schedule();
        finish_wait(wqh, &wait);
}

static int write_inode(struct inode *inode, struct writeback_control *wbc)
{
        int ret;

        if (inode->i_sb->s_op->write_inode && !is_bad_inode(inode)) {
//                trace_writeback_write_inode_start(inode, wbc);
                ret = inode->i_sb->s_op->write_inode(inode, wbc);
//                trace_writeback_write_inode(inode, wbc);
                return ret;
        }
        return 0;
}


/*
 * Handle writeback of dirty data for the device backed by this bdi. Also
 * reschedules periodically and does kupdated style flushing.
 */
void pxt4_wb_workfn(struct work_struct *work)
{
        struct bdi_writeback *wb = container_of(to_delayed_work(work),
                                                struct bdi_writeback, dwork);
        long pages_written; 

        set_worker_desc("flush-%s", bdi_dev_name(wb->bdi));

        if (likely(!current_is_workqueue_rescuer() ||
                   !test_bit(WB_registered, &wb->state))) {
                /*
                 * The normal path.  Keep writing back @wb until its
                 * work_list is empty.  Note that this path is also taken
                 * if @wb is shutting down even when we're running off the
                 * rescuer as work_list needs to be drained.
                 */
                do {
                        pages_written = pxt4_wb_do_writeback(wb);
                        // trace_writeback_pages_written(pages_written);
                } while (!list_empty(&wb->work_list));
        } else {
                /*
                 * bdi_wq can't get enough workers and we're running off
                 * the emergency worker.  Don't hog it.  Hopefully, 1024 is
                 * enough for efficient IO.
                 */
                pages_written = writeback_inodes_wb(wb, 1024,
                                                    WB_REASON_FORKER_THREAD);
                // trace_writeback_pages_written(pages_written);
        }

        if (!list_empty(&wb->work_list))
                wb_wakeup(wb);
        else if (wb_has_dirty_io(wb) && dirty_writeback_interval)
                wb_wakeup_delayed(wb);
}

