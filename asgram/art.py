# asgram/art.py
"""
Functions for making polished single image stereograms.
"""

from PIL import Image
import numpy as np
try:
    from asgram.utils import (
        _pixel_separation, _sirds_init, _asg_row, _conv_dots
    )
    from asgram.depth_map_making import ZMap
    from asgram.parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )
except ModuleNotFoundError:
    from utils import (
        _pixel_separation, _sirds_init, _asg_row, _conv_dots
    )
    from depth_map_making import ZMap
    from parallelize.local_process import (
        UPDATE_COUNTER, COUNTER, LOCK, STOP_EARLY,
        determine_processes, pool_runs
    )


def _p_worker(args):
    pull_from, zar, ref, mu, dpi, cross, approach, ys_to_build = args
    asg = np.zeros_like(pull_from)
    total = len(ys_to_build)
    prog = int(np.ceil(total / UPDATE_COUNTER))

    for i, y in enumerate(ys_to_build):
        asg[:, :, y] = _asg_row(
            pull_from, y, zar, ref, mu, dpi, cross, approach
        )

        if STOP_EARLY.is_set():
            break
        elif (i + 1) % prog == 0 or i + 1 == total:
            with LOCK:
                COUNTER.value += prog if (i + 1) % prog == 0 else total % prog

    return asg


def asgram(
    img, ref_img=None, ref_fit='fit',
    mu=1/3, dpi=72, cross=False, approach='rl',
    normalize=True, invert=False, smooth=False, pad=False, scale=1.0,
    palette='bw',
    dot_depth=0.0, dot_height='bottom',
    random_seed=1132,
    num_jobs=8
):
    """
    Creates an autostereogram from a depth (Z) map.

    Original Single Image Random Dot Stereogram algorithm described by
    Thimbleby, Inglis, & Witten (1994), adapted to Python.

    By default draws dots at the far plane. A depth value of 1
    will draw at the near plane. Values outside of [0, 1] will not draw.
    """
    np.random.seed(random_seed)
    _zmap = ZMap(img, mu, dpi, scale, smooth, smooth, invert, normalize, pad)
    zar = _zmap.zarr
    asg = _sirds_init(zar, ref_img, ref_fit, mu, dpi, approach, palette)

    jobs = determine_processes(num_jobs)
    if 1 < jobs:
        chunks = np.array_split(list(range(zar.shape[1])), jobs)
        _args = [
            (asg, zar, ref_img, mu, dpi, cross, approach, c)
            for c in chunks
        ]
        results = pool_runs(_p_worker, _args, zar.shape[1], jobs)
        asg = np.zeros_like(asg)
        for r in results:
            asg += r
    else:
        for y in range(zar.shape[1]):
            asg[:, :, y] = _asg_row(
                asg, y, zar, ref_img, mu, dpi, cross, approach
            )

    asg = _conv_dots(asg, dot_depth, dot_height, mu, dpi, cross)

    return Image.fromarray(asg.T)
