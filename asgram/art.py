# asgram/art.py
"""
Tool for making polished single image stereograms.
"""

try:
    from asgram.utils.params import Params
    from asgram.depth_map_making import ZMap
    from asgram.pixel_constraint_calculating import PixCon
    from asgram.source_pattern_making import SrcPat
    from asgram.postprocessing import Post
except ModuleNotFoundError:
    from utils.params import Params
    from depth_map_making import ZMap
    from pixel_constraint_calculating import PixCon
    from source_pattern_making import SrcPat
    from postprocessing import Post


def asgram(src, p: Params = Params(), ref=None):
    """
    Creates an autostereogram from a depth (Z) map.

    Original Single Image Random Dot Stereogram algorithm described by
    Thimbleby, Inglis, & Witten (1994), adapted to Python.
    """
    zm = ZMap(src, p)
    pc = PixCon(zm, p)
    sp = SrcPat(zm.size, pc, p, ref)
    asg = Post(sp, pc, p)
    return asg.final_img
