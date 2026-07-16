# asgram/parallelize/local_process.py
"""
Functions for parallelizing work in sgram using multiprocessing. Designed to be
used locally within notebooks for quick stereogram generation for example.
Based on the code from the farkle bot demo.
"""

import sys
import time
from multiprocessing import Pool, cpu_count, Value, Lock, Event
import numpy as np


# Animation
def progress_bar(i, total, start_time=time.perf_counter(), length=50):
    """
    Use during iteration to print progress updates to a task.
    """
    if total <= length:
        length = 10
    if total > length:
        loaded = int(round(length * i / total))
        prop_load = (i + 1) / total
        pct_load = int(round(prop_load * 100))
        time_diff = time.perf_counter() - start_time
        time_left = time_diff / prop_load * (1 - prop_load)
        min_left = int(np.floor(time_left / 60))
        sec_left = time_left % 60

        print_bar = "[" + "|" * loaded + "-" * (length - loaded) + "] "

        if pct_load < 10:
            print_bar += f"  {pct_load}% |"
        elif pct_load < 100:
            print_bar += f" {pct_load}% |"
        else:
            print_bar += f"{pct_load}% |"

        if time_diff < 5:
            print_bar += " ???m "
        elif min_left < 10:
            print_bar += f"   {min_left}m "
        elif min_left < 100:
            print_bar += f"  {min_left}m "
        elif min_left < 1000:
            print_bar += f" {min_left}m "
        else:
            print_bar += f"{min_left}m "

        if time_diff < 5:
            print_bar += "??.?s remaining  "
        elif sec_left < 10:
            print_bar += f" {sec_left:.1f}s remaining  "
        else:
            print_bar += f"{sec_left:.1f}s remaining  "

        if i + 1 == total:
            sys.stdout.write("\r" + " " * len(print_bar))
            sys.stdout.flush()
            sys.stdout.write("\r")
            sys.stdout.flush()
        else:
            sys.stdout.write("\r" + print_bar)
            sys.stdout.flush()


# Multiprocessing parameters
UPDATE_COUNTER = 1000
UPDATE_PROGRESS = 1.0


# Shared multiprocessing state, initialized per worker via Pool.initializer
COUNTER = None
LOCK = None
STOP_EARLY = None


def prog_init(shared_counter, shared_lock, stop_event):
    """
    Initializes shared variables to pass to workers during pooling.
    """
    # pylint: disable=global-statement
    global COUNTER, LOCK, STOP_EARLY
    COUNTER = shared_counter
    LOCK = shared_lock
    STOP_EARLY = stop_event


def determine_processes(processes=8):
    """
    Determine how many processes to run based on CPU core availability.
    """
    max_processes = cpu_count()
    if -1 == processes or processes > max_processes:
        processes = max_processes
    if 1 > processes:
        processes = 1
    return processes


def pool_runs(_func, args_list, num_sims, processes):
    """
    Generic framework for parallelizing iterative processes using
    multiprocessing. Also allows for early stopping, and prints progress as
    dynamic one-line text.
    """
    # pylint: disable=global-statement
    global COUNTER, LOCK, STOP_EARLY
    COUNTER = Value('i', 0)
    LOCK = Lock()
    STOP_EARLY = Event()

    with Pool(
        processes=processes,
        initializer=prog_init,
        initargs=(COUNTER, LOCK, STOP_EARLY)
    ) as pool:
        async_results = pool.map_async(_func, args_list)

        start_time = time.perf_counter()
        prev_completed = -1
        try:
            while not async_results.ready():
                with COUNTER.get_lock():
                    completed = COUNTER.value
                    if completed > prev_completed:
                        progress_bar(completed, num_sims, start_time)
                        prev_completed = completed
                time.sleep(UPDATE_PROGRESS)
        except KeyboardInterrupt:
            STOP_EARLY.set()
            print("\nstopping early...")
            async_results.wait(timeout=5)
            pool.terminate()
            pool.join()

        results = async_results.get()
        progress_bar(num_sims - 1, num_sims)

    return results
