# asgram/utils/progress_bar.py
"""
Method which prints a progress bar to sys.stdout.
"""

import sys
import time
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
