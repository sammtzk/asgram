# asgram/art.py
"""
Functions for making polished single image stereograms.
"""

import numpy as np
from PIL import Image
try:
    from asgram.algorithm import _dsdsc
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.postprocessing import finish
    from asgram.utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )
except ModuleNotFoundError:
    from algorithm import _dsdsc
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from postprocessing import finish
    from utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )


def _synth_worker(_args):
    """Worker for synthesizer parallelization. Wraps generic run_worker."""
    ys_to_build, args_dict = _args

    def _row_func_wrapper(y, ad=args_dict):
        """Wrapper for _dsdsc."""
        return _dsdsc(
            y=y,
            zar=ad['src_mat'],
            _re=ad['_re'],
            mu=ad['mu'],
            dpi=ad['dpi'],
            cross_eyed=ad['cross'],
            approach=ad['approach']
        )

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, 2, np.uint16)


def synthesizer(
        zmap: ZMap, sp: SrcPat,
        mu=1/3, dpi=72, cross=False, approach='rl',
        num_jobs=8
):
    """Creates an autostereogram using params, a ZMap, and a SourcePattern."""
    print("Step: Constraint Building")
    zar = zmap.zm_arr
    _re = sp.ref is not None

    jobs = worker_count(num_jobs)
    if 1 < jobs:
        args_dict = {
            'src_mat': zar,
            'total_ys': zar.shape[1],
            '_re': _re,
            'mu': mu,
            'dpi': dpi,
            'cross': cross,
            'approach': approach
        }
        con = parallelize_workers(args_dict, _synth_worker, jobs, np.uint16)
    else:
        con = np.zeros_like(zar, dtype=np.uint16)
        for y in range(zar.shape[1]):
            con[:, y] = _dsdsc(y, zar, _re, mu, dpi, cross, approach)

    asg = np.take_along_axis(sp.sp_arr, con[None, :, :], axis=1)

    print("Complete.")
    return asg


def asgram(
    src, ref=None, rfit='fit',
    mu=1/3, dpi=72, cross=False, approach='rl',
    normalize=True, invert=False, iis=False, bil=False, pad=False, scale=1.0,
    rpal='bw', rseed=1132,
    pdvrs=False, dot_depth=0.0, dot_height='bottom',
    num_jobs=8
):
    """
    Creates an autostereogram from a depth (Z) map.

    Original Single Image Random Dot Stereogram algorithm described by
    Thimbleby, Inglis, & Witten (1994), adapted to Python.
    """
    _zm = ZMap(src, mu, dpi, scale, iis, bil, invert, normalize, pad, num_jobs)
    _sp = SrcPat(_zm.size, ref, cross, mu, dpi, rfit, approach, rpal, rseed)
    asg = synthesizer(_zm, _sp, mu, dpi, cross, approach, num_jobs)
    asg = finish(asg, dot_depth, dot_height, mu, dpi, cross, pdvrs, num_jobs)

    return Image.fromarray(asg.T)
