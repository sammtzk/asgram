# asgram/art.py
"""
Tool for making polished single image stereograms.
"""

try:
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.pixel_constraint_calculating import PixCon
    from asgram.postprocessing import Post
except ModuleNotFoundError:
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from pixel_constraint_calculating import PixCon
    from postprocessing import Post


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
    _pc = PixCon(_zm, _sp, mu, dpi, cross, approach, num_jobs)
    asg = Post(_pc, dot_depth, dot_height, mu, dpi, cross, pdvrs, num_jobs)
    return asg.final_img
