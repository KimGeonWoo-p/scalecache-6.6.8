#include <linux/kernel.h>
#include <linux/math64.h>
#include <linux/export.h>
#include <linux/spinlock.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/swap.h>
#include <linux/slab.h>
#include <linux/pagemap.h>
#include <linux/writeback.h>
#include <linux/init.h>
#include <linux/backing-dev.h>
#include <linux/task_io_accounting_ops.h>
#include <linux/blkdev.h>
#include <linux/mpage.h>
#include <linux/rmap.h>
#include <linux/percpu.h>
#include <linux/smp.h>
#include <linux/sysctl.h>
#include <linux/cpu.h>
#include <linux/syscalls.h>
#include <linux/pagevec.h>
#include <linux/timer.h>
#include <linux/sched/rt.h>
#include <linux/sched/signal.h>
#include <linux/mm_inline.h>
#include <trace/events/writeback.h>

#include "internal.h"












struct dirty_throttle_control {
#ifdef CONFIG_CGROUP_WRITEBACK
	struct wb_domain	*dom;
	struct dirty_throttle_control *gdtc;	/* only set in memcg dtc's */
#endif
	struct bdi_writeback	*wb;
	struct fprop_local_percpu *wb_completions;

	unsigned long		avail;		/* dirtyable */
	unsigned long		dirty;		/* file_dirty + write + nfs */
	unsigned long		thresh;		/* dirty threshold */
	unsigned long		bg_thresh;	/* dirty background threshold */

	unsigned long		wb_dirty;	/* per-wb counterparts */
	unsigned long		wb_thresh;
	unsigned long		wb_bg_thresh;

	unsigned long		pos_ratio;
};


#define GDTC_INIT(__wb)		.wb = (__wb),				\
				.dom = &global_wb_domain,		\
				.wb_completions = &(__wb)->completions

#define GDTC_INIT_NO_WB		.dom = &global_wb_domain

#define MDTC_INIT(__wb, __gdtc)	.wb = (__wb),				\
				.dom = mem_cgroup_wb_domain(__wb),	\
				.wb_completions = &(__wb)->memcg_completions, \
				.gdtc = __gdtc

#define BANDWIDTH_INTERVAL	max(HZ/5, 1)
#define RATELIMIT_CALC_SHIFT	10

static bool mdtc_valid(struct dirty_throttle_control *dtc)
{
	return dtc->dom;
}

unsigned long global_dirtyable_memory(void);
void domain_dirty_limits(struct dirty_throttle_control *dtc);
inline void wb_dirty_limits(struct dirty_throttle_control *dtc);
void mdtc_calc_avail(struct dirty_throttle_control *mdtc,
			    unsigned long filepages, unsigned long headroom);
unsigned long dirty_freerun_ceiling(unsigned long thresh,
					   unsigned long bg_thresh);
unsigned long dirty_poll_interval(unsigned long dirty,
					 unsigned long thresh);
void wb_position_ratio(struct dirty_throttle_control *dtc);
void __wb_update_bandwidth(struct dirty_throttle_control *gdtc,
				  struct dirty_throttle_control *mdtc,
				  bool update_ratelimit);
unsigned long wb_max_pause(struct bdi_writeback *wb,
				  unsigned long wb_dirty);
long wb_min_pause(struct bdi_writeback *wb,
			 long max_pause,
			 unsigned long task_ratelimit,
			 unsigned long dirty_ratelimit,
			 int *nr_dirtied_pause);
long wb_do_writeback_modified(struct bdi_writeback *wb);

int df_balance_dirty_pages(struct bdi_writeback *wb,
			       unsigned long pages_dirtied, unsigned int flags)
{
	struct dirty_throttle_control gdtc_stor = { GDTC_INIT(wb) };
	struct dirty_throttle_control mdtc_stor = { MDTC_INIT(wb, &gdtc_stor) };
	struct dirty_throttle_control * const gdtc = &gdtc_stor;
	struct dirty_throttle_control * const mdtc = mdtc_valid(&mdtc_stor) ?
						     &mdtc_stor : NULL;
	struct dirty_throttle_control *sdtc;
	unsigned long nr_reclaimable;	/* = file_dirty */
	long period;
	long pause;
	long max_pause;
	long min_pause;
	int nr_dirtied_pause;
	bool dirty_exceeded = false;
	unsigned long task_ratelimit;
	unsigned long dirty_ratelimit;
	struct backing_dev_info *bdi = wb->bdi;
	bool strictlimit = bdi->capabilities & BDI_CAP_STRICTLIMIT;
	unsigned long start_time = jiffies;
	int ret = 0;

	for (;;) {
		unsigned long now = jiffies;
		unsigned long dirty, thresh, bg_thresh;
		unsigned long m_dirty = 0;	/* stop bogus uninit warnings */
		unsigned long m_thresh = 0;
		unsigned long m_bg_thresh = 0;

		nr_reclaimable = global_node_page_state(NR_FILE_DIRTY);
		gdtc->avail = global_dirtyable_memory();
		gdtc->dirty = nr_reclaimable + global_node_page_state(NR_WRITEBACK);

		domain_dirty_limits(gdtc);

		if (unlikely(strictlimit)) {
			wb_dirty_limits(gdtc);

			dirty = gdtc->wb_dirty;
			thresh = gdtc->wb_thresh;
			bg_thresh = gdtc->wb_bg_thresh;
		} else {
			dirty = gdtc->dirty;
			thresh = gdtc->thresh;
			bg_thresh = gdtc->bg_thresh;
		}

		if (mdtc) {
			unsigned long filepages, headroom, writeback;

			/*
			 * If @wb belongs to !root memcg, repeat the same
			 * basic calculations for the memcg domain.
			 */
			mem_cgroup_wb_stats(wb, &filepages, &headroom,
					    &mdtc->dirty, &writeback);
			mdtc->dirty += writeback;
			mdtc_calc_avail(mdtc, filepages, headroom);

			domain_dirty_limits(mdtc);

			if (unlikely(strictlimit)) {
				wb_dirty_limits(mdtc);
				m_dirty = mdtc->wb_dirty;
				m_thresh = mdtc->wb_thresh;
				m_bg_thresh = mdtc->wb_bg_thresh;
			} else {
				m_dirty = mdtc->dirty;
				m_thresh = mdtc->thresh;
				m_bg_thresh = mdtc->bg_thresh;
			}
		}

		/*
		 * In laptop mode, we wait until hitting the higher threshold
		 * before starting background writeout, and then write out all
		 * the way down to the lower threshold.  So slow writers cause
		 * minimal disk activity.
		 *
		 * In normal mode, we start background writeout at the lower
		 * background_thresh, to keep the amount of dirty memory low.
		 */
		if (!laptop_mode && nr_reclaimable > gdtc->bg_thresh &&
		    !writeback_in_progress(wb))
			wb_start_background_writeback(wb);

		/*
		 * Throttle it only when the background writeback cannot
		 * catch-up. This avoids (excessively) small writeouts
		 * when the wb limits are ramping up in case of !strictlimit.
		 *
		 * In strictlimit case make decision based on the wb counters
		 * and limits. Small writeouts when the wb limits are ramping
		 * up are the price we consciously pay for strictlimit-ing.
		 *
		 * If memcg domain is in effect, @dirty should be under
		 * both global and memcg freerun ceilings.
		 */
		if (dirty <= dirty_freerun_ceiling(thresh, bg_thresh) &&
		    (!mdtc ||
		     m_dirty <= dirty_freerun_ceiling(m_thresh, m_bg_thresh))) {
			unsigned long intv;
			unsigned long m_intv;

free_running:
			intv = dirty_poll_interval(dirty, thresh);
			m_intv = ULONG_MAX;

			current->dirty_paused_when = now;
			current->nr_dirtied = 0;
			if (mdtc)
				m_intv = dirty_poll_interval(m_dirty, m_thresh);
			current->nr_dirtied_pause = min(intv, m_intv);
			break;
		}

		/* Start writeback even when in laptop mode */
		if (unlikely(!writeback_in_progress(wb)))
			wb_start_background_writeback(wb);

		mem_cgroup_flush_foreign(wb);

		/*
		 * Calculate global domain's pos_ratio and select the
		 * global dtc by default.
		 */
		if (!strictlimit) {
			wb_dirty_limits(gdtc);

			if ((current->flags & PF_LOCAL_THROTTLE) &&
			    gdtc->wb_dirty <
			    dirty_freerun_ceiling(gdtc->wb_thresh,
						  gdtc->wb_bg_thresh))
				/*
				 * LOCAL_THROTTLE tasks must not be throttled
				 * when below the per-wb freerun ceiling.
				 */
				goto free_running;
		}

		dirty_exceeded = (gdtc->wb_dirty > gdtc->wb_thresh) &&
			((gdtc->dirty > gdtc->thresh) || strictlimit);

		wb_position_ratio(gdtc);
		sdtc = gdtc;

		if (mdtc) {
			/*
			 * If memcg domain is in effect, calculate its
			 * pos_ratio.  @wb should satisfy constraints from
			 * both global and memcg domains.  Choose the one
			 * w/ lower pos_ratio.
			 */
			if (!strictlimit) {
				wb_dirty_limits(mdtc);

				if ((current->flags & PF_LOCAL_THROTTLE) &&
				    mdtc->wb_dirty <
				    dirty_freerun_ceiling(mdtc->wb_thresh,
							  mdtc->wb_bg_thresh))
					/*
					 * LOCAL_THROTTLE tasks must not be
					 * throttled when below the per-wb
					 * freerun ceiling.
					 */
					goto free_running;
			}
			dirty_exceeded |= (mdtc->wb_dirty > mdtc->wb_thresh) &&
				((mdtc->dirty > mdtc->thresh) || strictlimit);

			wb_position_ratio(mdtc);
			if (mdtc->pos_ratio < gdtc->pos_ratio)
				sdtc = mdtc;
		}

		if (dirty_exceeded != wb->dirty_exceeded)
			wb->dirty_exceeded = dirty_exceeded;

		if (time_is_before_jiffies(READ_ONCE(wb->bw_time_stamp) +
					   BANDWIDTH_INTERVAL))
			__wb_update_bandwidth(gdtc, mdtc, true);

		/* throttle according to the chosen dtc */
		dirty_ratelimit = READ_ONCE(wb->dirty_ratelimit);
		task_ratelimit = ((u64)dirty_ratelimit * sdtc->pos_ratio) >>
							RATELIMIT_CALC_SHIFT;
		max_pause = wb_max_pause(wb, sdtc->wb_dirty);
		min_pause = wb_min_pause(wb, max_pause,
					 task_ratelimit, dirty_ratelimit,
					 &nr_dirtied_pause);

		if (unlikely(task_ratelimit == 0)) {
			period = max_pause;
			pause = max_pause;
			goto pause;
		}
		period = HZ * pages_dirtied / task_ratelimit;
		pause = period;
		if (current->dirty_paused_when)
			pause -= now - current->dirty_paused_when;
		/*
		 * For less than 1s think time (ext3/4 may block the dirtier
		 * for up to 800ms from time to time on 1-HDD; so does xfs,
		 * however at much less frequency), try to compensate it in
		 * future periods by updating the virtual time; otherwise just
		 * do a reset, as it may be a light dirtier.
		 */
		if (pause < min_pause) {
			trace_balance_dirty_pages(wb,
						  sdtc->thresh,
						  sdtc->bg_thresh,
						  sdtc->dirty,
						  sdtc->wb_thresh,
						  sdtc->wb_dirty,
						  dirty_ratelimit,
						  task_ratelimit,
						  pages_dirtied,
						  period,
						  min(pause, 0L),
						  start_time);
			if (pause < -HZ) {
				current->dirty_paused_when = now;
				current->nr_dirtied = 0;
			} else if (period) {
				current->dirty_paused_when += period;
				current->nr_dirtied = 0;
			} else if (current->nr_dirtied_pause <= pages_dirtied)
				current->nr_dirtied_pause += pages_dirtied;
			break;
		}
		if (unlikely(pause > max_pause)) {
			/* for occasional dropped task_ratelimit */
			now += min(pause - max_pause, max_pause);
			pause = max_pause;
		}

pause:
		trace_balance_dirty_pages(wb,
					  sdtc->thresh,
					  sdtc->bg_thresh,
					  sdtc->dirty,
					  sdtc->wb_thresh,
					  sdtc->wb_dirty,
					  dirty_ratelimit,
					  task_ratelimit,
					  pages_dirtied,
					  period,
					  pause,
					  start_time);
		if (flags & BDP_ASYNC) {
			ret = -EAGAIN;
			break;
		}
		__set_current_state(TASK_KILLABLE);
		wb->dirty_sleep = now;
		io_schedule_timeout(pause);

		wb_do_writeback_modified(wb);

		current->dirty_paused_when = now + pause;
		current->nr_dirtied = 0;
		current->nr_dirtied_pause = nr_dirtied_pause;

		/*
		 * This is typically equal to (dirty < thresh) and can also
		 * keep "1000+ dd on a slow USB stick" under control.
		 */
		if (task_ratelimit)
			break;

		/*
		 * In the case of an unresponsive NFS server and the NFS dirty
		 * pages exceeds dirty_thresh, give the other good wb's a pipe
		 * to go through, so that tasks on them still remain responsive.
		 *
		 * In theory 1 page is enough to keep the consumer-producer
		 * pipe going: the flusher cleans 1 page => the task dirties 1
		 * more page. However wb_dirty has accounting errors.  So use
		 * the larger and more IO friendly wb_stat_error.
		 */
		if (sdtc->wb_dirty <= wb_stat_error())
			break;

		if (fatal_signal_pending(current))
			break;
	}
	return ret;
}

