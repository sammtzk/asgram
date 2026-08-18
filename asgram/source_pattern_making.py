# asgram/source_pattern_making.py
"""
Functions for building, modifying, and generating source patterns.
"""

import numpy as np
from PIL import Image
from matplotlib import colormaps
try:
    from asgram.utils.utils import _pixel_separation
except ModuleNotFoundError:
    from utils.utils import _pixel_separation


# Pattern Making Helpers
def enforce_source(ref, w, h, source_width, approach='rl'):
    """
    Ensures that unconstrained asgram pixels also originate from
    approach-specific sources by isolating the source image.
    """
    ref = ref.resize((w, h))
    asg = np.array(ref.convert('RGB')).T

    match approach:
        case 'mo':
            lbound = round((w - source_width) / 2)
            rbound = source_width + lbound
            source = asg[:, lbound:rbound]
        case 'lr':
            source = asg[:, :source_width]
        case _:     # rl as default
            source = asg[:, -source_width:]

    return Image.fromarray(source.T)


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


# Special Cases (oi source specification, rds)
def _enforce_oi_source(ref, w, h, source_width):
    """Handles the special case of outer source pixels."""
    ref = ref.resize((w, h))
    asg = np.array(ref.convert('RGB')).T

    lwidth = round(w / 2)
    rwidth = w - lwidth

    lsource = asg[:, :source_width]
    rsource = asg[:, -source_width:]

    lasg = np.tile(lsource, (1, _rotf(lwidth, source_width), 1))[:, :lwidth]
    rasg = np.tile(rsource, (1, _rotf(lwidth, source_width), 1))[:, -rwidth:]

    return np.concat([lasg, rasg], axis=1)


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

    def __init__(
            self, size, ref=None, cross_eyed=False,
            mu=1/3, dpi=72, fit='fit', approach='rl',
            random_palette='bw'
    ):
        self.size = size
        self.ref = ref
        self.cross_eyed = cross_eyed

        self.mu = mu
        self.dpi = dpi
        self.fit = fit
        self.approach = approach

        self.random_palette = random_palette

        self.sp_arr = np.array([])
        self.sp_img = Image.new('1', (0, 0))
        self.update()

    def update(self):
        """Updates the source pattern according to class parameters."""
        print("Step: Source Pattern Making")
        if self.ref is not None:
            asg = self.ref.copy()
            w, h = self.size
            rep_len = _pixel_separation(0, self.mu, self.dpi, self.cross_eyed)

            if ('ES' == self.fit) and ('oi' == self.approach):
                asg = _enforce_oi_source(asg, w, h, rep_len)
            elif ('ES' == self.fit) and ('random' != self.approach):
                asg = enforce_source(asg, w, h, rep_len, self.approach)
                asg = asgram_tiler(asg, w, h, rep_len, 'htile')
            else:
                asg = asgram_tiler(asg, w, h, rep_len, self.fit)
            asg = source_crop(asg, w, h, self.approach)

        else:
            _col_pal = _color_palette_maker(self.random_palette)
            asg = np.array(
                _col_pal[np.random.randint(len(_col_pal), size=self.size)],
                dtype=np.uint8
            ).transpose(2, 0, 1)

        self.sp_arr = asg
        self.sp_img = Image.fromarray(asg.T)
        print("Complete.")
