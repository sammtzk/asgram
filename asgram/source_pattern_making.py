# asgram/source_pattern_making.py
"""
Functions for building, modifying, and generating source patterns.
"""

import numpy as np
from matplotlib import colormaps
try:
    from asgram.algorithm import _pixel_separation
except ModuleNotFoundError:
    from algorithm import _pixel_separation


def _color_palette_maker(palette='bw'):
    if palette in list(colormaps):
        color_palette = colormaps[palette](np.linspace(0, 1, 8))
        color_palette = np.round(color_palette[:, :3] * 255).astype('uint8')
    else:
        color_palette = np.array([(0, 0, 0), (255, 255, 255)], np.uint8)

    return color_palette


def _pattern_maker(size, ce, ref, fit='fit', mu=1/3, dpi=72, approach='rl'):
    w, h = size
    w_rep_len, h_rep_len = None, None
    repeat_len = _pixel_separation(0, mu, dpi, cross_eyed=ce)

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
            ref = ref.copy().resize((w_rep_len, h_rep_len))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, w // w_rep_len + 2, h // h_rep_len + 2))
        case 'htile':
            ref = ref.copy().resize((repeat_len, h))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, w // repeat_len + 2, 1))
        case 'vtile':
            ref = ref.copy().resize((w, repeat_len))
            asg = np.array(ref.convert('RGB')).T
            asg = np.tile(asg, (1, 1, h // repeat_len + 2))
        case _:     # fit as default
            ref = ref.copy().resize((w, h))
            asg = np.array(ref.convert('RGB')).T

    if fit != 'fit':
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


def _sirds_init(
    zar, ce, ref=None, fit='fit', mu=1/3, dpi=72, approach='rl', palette='bw'
):
    if ref is not None:
        asg = _pattern_maker(zar.shape, ce, ref, fit, mu, dpi, approach)
    else:
        color_pal = _color_palette_maker(palette)
        asg = np.array(
            color_pal[np.random.randint(len(color_pal), size=zar.shape)],
            dtype=np.uint8
        ).transpose(2, 0, 1)

    return asg
