# asgram/utils/parallelize.py
"""
Generic framework for parallelization of expensive processes.
"""

import time
from multiprocessing import cpu_count, get_context
import numpy as np
try:
    from asgram.utils.progress_bar import progress_bar
except ModuleNotFoundError:
    from utils.progress_bar import progress_bar


# Multiprocessing parameters
UPDATE_COUNTER = 1000
UPDATE_PROGRESS = 1.0


# Generic parallelization framework
def worker_count(processes=8):
    """Determine how many processes to run based on CPU core availability."""
    max_processes = cpu_count()
    if -1 == processes or processes > max_processes:
        processes = max_processes
    if 1 > processes:
        processes = 1
    return processes


def worker_init(shared_counter, shared_lock, shared_stop_event):
    """Initialize shared multiprocessing state across workers."""
    # pylint: disable=global-statement
    global COUNTER, LOCK, STOP_EARLY
    COUNTER = shared_counter
    LOCK = shared_lock
    STOP_EARLY = shared_stop_event


# necessary keys for args_dict:
# src_mat (source matrix of the same shape as the output matrix)
# total_ys (total number of output matrix rows)


def run_worker(ys_to_build, args_dict, spec_row_func, dim=3):
    """Handle parallelization states for specialized workers."""
    total = len(ys_to_build)
    prog = int(np.ceil(total / UPDATE_COUNTER))
    build_mat = np.zeros_like(args_dict['src_mat'])

    for i, y, in enumerate(ys_to_build):
        if dim == 2:
            build_mat[:, y] = spec_row_func(y, args_dict)
        else:
            build_mat[:, :, y] = spec_row_func(y, args_dict)

        if STOP_EARLY.is_set():
            break
        elif (i + 1) % prog == 0 or i + 1 == total:
            with LOCK:
                COUNTER.value += prog if (i + 1) % prog == 0 else total % prog

    return build_mat


def parallelize_workers(args_dict, spec_worker_func, processes):
    """
    Generic framework for parallelizing iterative matrix-building processes
    using multiprocessing. Also allows for early stopping, and prints progress
    as dynamic one-line text.
    """
    total_ys = args_dict['total_ys']

    start_time = time.perf_counter()
    progress_bar(0, total_ys, start_time)

    chunks = np.array_split(list(range(total_ys)), processes)
    _args = [(c, args_dict) for c in chunks]

    par_ctx = get_context('spawn')
    with par_ctx.Manager() as manager:
        COUNTER = manager.Value('i', 0)
        LOCK = manager.Lock()
        STOP_EARLY = manager.Event()

        with par_ctx.Pool(
            processes=processes,
            initializer=worker_init,
            initargs=(COUNTER, LOCK, STOP_EARLY)
        ) as pool:
            async_results = pool.map_async(spec_worker_func, _args)

            prev_completed = -1
            try:
                while not async_results.ready():
                    completed = COUNTER.value
                    if completed > prev_completed:
                        progress_bar(completed, total_ys, start_time)
                        prev_completed = completed
                    time.sleep(UPDATE_PROGRESS)
            except KeyboardInterrupt:
                STOP_EARLY.set()
                print("\nstopping early...")
                async_results.wait(timeout=5)
                pool.terminate()
                pool.join()

            results = async_results.get()
            progress_bar(total_ys - 1, total_ys)

    output_matrix = np.zeros_like(args_dict['src_mat'])
    for r in results:
        if r is not None:
            output_matrix += r
    return output_matrix
