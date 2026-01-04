# sgram/parallelize/gradio_process.py
"""
Docstring or smth.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, set_start_method
import numpy as np


EXECUTOR = None


def set_processes(processes=5, concurrency_limit=4):
    """
    Determine how many processes to run based on CPU core availability.
    """
    max_processes = np.floor(cpu_count() / concurrency_limit).item()
    if -1 == processes or processes > max_processes:
        processes = max_processes
    if 1 > processes:
        processes = 1
    return processes


def executor_init(max_workers):
    """
    Initializes pool executor.
    """
    # pylint: disable=global-statement
    global EXECUTOR
    if EXECUTOR is None:
        set_start_method('spawn', force=True)
        EXECUTOR = ProcessPoolExecutor(max_workers)


async def executor_runs(_func, args_list):
    """
    Simple framework for parallelizing iterative processes using
    multiprocessing. Does not provide detail. Works with gradio.
    """
    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(EXECUTOR, _func, _a) for _a in args_list]

    results = []
    while futures:
        done, futures = await asyncio.wait(
            futures, return_when=asyncio.FIRST_COMPLETED
        )
        for f in done:
            results.append(await f)
        await asyncio.sleep(0.05)

    return results
