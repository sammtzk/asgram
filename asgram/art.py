# asgram/art.py
"""
Functions for making polished single image stereograms.
"""

import numpy as np
from PIL import Image
try:
    from asgram.algorithm import _asg_row
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.postprocessing import finish
    from asgram.parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )
except ModuleNotFoundError:
    from algorithm import _asg_row
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from postprocessing import finish
    from parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )


def _asgram_row_worker(args):
    pull_from, zar, _re, mu, dpi, cross, approach, ys_to_build = args
    asg = np.zeros_like(pull_from)
    total = len(ys_to_build)
    prog = int(np.ceil(total / UPDATE_COUNTER))

    for i, y in enumerate(ys_to_build):
        asg[:, :, y] = _asg_row(
            pull_from, y, zar, _re, mu, dpi, cross, approach
        )

        if STOP_EARLY.is_set():
            break
        elif (i + 1) % prog == 0 or i + 1 == total:
            with LOCK:
                COUNTER.value += prog if (i + 1) % prog == 0 else total % prog

    return asg


def synthesizer(
        zmap: ZMap, sp: SrcPat,
        mu=1/3, dpi=72, cross=False, approach='rl',
        num_jobs=8
):
    """Creates an autostereogram using params, a ZMap, and a SourcePattern."""
    print("Step: Constraint Building")
    zar = zmap.zm_arr
    _re = sp.ref is not None
    asg = sp.sp_arr

    jobs = determine_processes(num_jobs)
    if 1 < jobs:
        chunks = np.array_split(list(range(zar.shape[1])), jobs)
        _args = [
            (asg, zar, _re, mu, dpi, cross, approach, c)
            for c in chunks
        ]
        results = pool_runs(_asgram_row_worker, _args, zar.shape[1], jobs)
        asg = np.zeros_like(asg)
        for r in results:
            asg += r
    else:
        for y in range(zar.shape[1]):
            asg[:, :, y] = _asg_row(
                asg, y, zar, _re, mu, dpi, cross, approach
            )

    print("Complete.")
    return asg


def asgram(
    img, ref=None, rfit='fit',
    mu=1/3, dpi=72, cross=False, approach='rl',
    normalize=True, invert=False, iis=False, bil=False, pad=False, scale=1.0,
    rpal='bw', random_seed=1132,
    pdvrs=False, dot_depth=0.0, dot_height='bottom',
    num_jobs=8
):
    """
    Creates an autostereogram from a depth (Z) map.

    Original Single Image Random Dot Stereogram algorithm described by
    Thimbleby, Inglis, & Witten (1994), adapted to Python.
    """
    np.random.seed(random_seed)
    _zm = ZMap(img, mu, dpi, scale, iis, bil, invert, normalize, pad, num_jobs)
    _sp = SrcPat(_zm.size, ref, cross, mu, dpi, rfit, approach, rpal)
    asg = synthesizer(_zm, _sp, mu, dpi, cross, approach, num_jobs)
    asg = finish(asg, dot_depth, dot_height, mu, dpi, cross, pdvrs, num_jobs)

    return Image.fromarray(asg.T)
