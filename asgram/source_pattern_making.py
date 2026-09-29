# asgram/source_pattern_making.py
"""
Functions for building, modifying, and generating source patterns.
"""

from typing import Union
import numpy as np
from PIL import Image
import cv2 as cv
from matplotlib import colormaps
try:
    from asgram.utils.utils import _pixel_separation as _pix_sep
    from asgram.utils.params import Params
    from asgram.depth_map_making import ZMap
    from asgram.pixel_constraint_calculating import PixCon
except ModuleNotFoundError:
    from utils.utils import _pixel_separation as _pix_sep
    from utils.params import Params
    from depth_map_making import ZMap
    from pixel_constraint_calculating import PixCon


# Pattern Making Helpers
def _rotf(whole, part):
    """Repeat (Odd) Times Finder"""
    _rep_times = int(np.ceil(whole / part))
    return _rep_times if _rep_times % 2 else _rep_times + 1


def asgram_tiler(ref, w, h, repeat_len, fit):
    """Uses reference image to tile asgram source patterns."""
    w_rep_len = h_rep_len = None
    if any(ch.isnumeric() for ch in fit):
        # expect fit='tile=(w_reps)x(h_reps)'
        try:
            w_reps, h_reps = [float(c) for c in fit[5:].split('x')]
            w_rep_len = int(round(w / w_reps))
            h_rep_len = int(round(h / h_reps))
            fit = 'tile'
        except ValueError:
            fit = 'tile'

    match fit:
        case 'tile':
            if w_rep_len is None or h_rep_len is None:
                w_rep_len = h_rep_len = repeat_len
            ref = ref.resize((w_rep_len, h_rep_len))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, _rotf(w, w_rep_len), _rotf(h, h_rep_len)))
        case 'htile':
            ref = ref.resize((repeat_len, h))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, _rotf(w, repeat_len), 1))
        case 'vtile':
            ref = ref.resize((w, repeat_len))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, 1, _rotf(h, repeat_len)))
        case _:     # fit as default
            ref = ref.resize((w, h))
            asg = np.array(ref.convert('RGB')).T

    return asg


def source_crop(asg, w, h, approach):
    """Crops excess of asgram pattern to match dimensions of depth map."""
    lower = round((asg.shape[2] - h) / 2)
    upper = h + lower

    match approach:
        case 'mo' | 'oi' | 'random':
            lbound = round((asg.shape[1] - w) / 2)
            rbound = w + lbound
            asg = asg[:, lbound:rbound, lower:upper]
        case 'lr':
            asg = asg[:, :w, lower:upper]
        case _:     # rl as default
            asg = asg[:, -w:, lower:upper]

    return asg


def _color_palette_maker(palette='bw'):
    if palette in list(colormaps):
        color_palette = colormaps[palette](np.linspace(0, 1, 8))
        color_palette = np.round(color_palette[:, :3] * 255).astype('uint8')
    else:
        color_palette = np.array([(0, 0, 0), (255, 255, 255)], np.uint8)

    return color_palette


# Object Maker
class SrcPat:
    """Stores and augments source patterns for autostereograms."""

    def __init__(self, zm: ZMap, pc: Union[PixCon, None], p: Params, ref=None):
        self.zm = zm
        self.pc = pc
        self.p = p
        self.ref = ref

        self.sp_arr = np.array([])
        self.sp_img = Image.new('1', (0, 0))
        self.update()

    @property
    def size(self):
        return self.zm.size

    def _fit_to_source(self, asg):
        """Refits the asgram pattern to match the PixCon source area."""
        if self.p.source_fit in ['estimate', 'exact']:
            pixcon_dummy = self.pc
            if pixcon_dummy is None:
                pixcon_dummy = PixCon(
                    zmap=self.zm,
                    p=self.p,
                    update_on_init=False
                )

            if 'exact' == self.p.source_fit:
                src = pixcon_dummy.src_area
            else:
                src = pixcon_dummy.src_area_basic
            height = src.shape[1]
            assert asg.shape[2] == height

            input = asg.T
            output = np.zeros_like(asg, dtype=np.uint8)

            for y in range(height):
                src_indices = np.where(src[:, y])[0]
                src_width = len(src_indices)
                src_fit = cv.resize(
                    input[[y]], (src_width, 1),
                    interpolation=cv.INTER_LANCZOS4
                )[0].T
                output[:, src_indices, y] = src_fit
            asg = output

        return asg

    def update(self):
        """Updates the source pattern according to class parameters."""
        print("Step: Source Pattern Making")
        np.random.seed(self.p.random_seed)
        if self.ref is not None:
            asg = self.ref.copy()
            w, h = self.size
            rep_len = _pix_sep(0, self.p.mu, self.p.dpi, self.p.cross)

            asg = asgram_tiler(asg, w, h, rep_len, self.p.pattern_fit)
            asg = source_crop(asg, w, h, self.p.approach)
            asg = self._fit_to_source(asg)

        else:
            _col_pal = _color_palette_maker(self.p.random_pattern_palette)
            asg = np.array(
                _col_pal[np.random.randint(len(_col_pal), size=self.size)],
                dtype=np.uint8
            ).transpose(2, 0, 1)

        self.sp_arr = asg
        self.sp_img = Image.fromarray(asg.T).convert('RGB')
        print("Complete.")

    def save(self, file_name='temp', dir_path='', extension='.png'):
        """
        Saves pattern as an image of a specified format.

        Extension should be one of '.png', '.jpg', or '.jpeg'.
        """
        dir_path = dir_path + '/' if dir_path else dir_path
        file_path = dir_path + file_name + extension

        if extension in ('.png', '.jpg', '.jpeg'):
            with open(file_path, 'wb') as f:
                self.sp_img.save(f)
