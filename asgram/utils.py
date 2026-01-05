# sgram/utils.py
"""
Utility functions for the creation of autostereograms.
"""

import numpy as np
from matplotlib import colormaps
try:
    from asgram.tiw import _do_work, _separation
except ModuleNotFoundError:
    from tiw import _do_work, _separation


def _pixel_separation(Z, mu=1/3, dpi=72, cross_eyed=False):
    Z = -Z + 1.0 if cross_eyed else Z
    return _separation(Z=Z, mu=mu, dpi=dpi)


def _resize_img(_img, mult=1.0):
    if 0.0 < mult and 1.0 != mult:
        _img = _img.resize([int(round(n * np.sqrt(mult))) for n in _img.size])
    return _img


def _invert_img_array(_arr):
    return ((_arr.astype('int16') - 255) * -1).astype('uint8')


def _normalize_img_array(_arr, normalize):
    _arr = np.array(_arr, dtype=np.float32)
    if not normalize:
        return _arr / 255
    return (_arr - np.min(_arr)) / (np.max(_arr) - np.min(_arr))


def _pad_img_array(_arr, mu=1/3, dpi=72):
    far = _pixel_separation(0, mu, dpi, cross_eyed=False)
    l_pad = np.repeat(_arr[0:1, :], far, axis=0)
    r_pad = np.repeat(_arr[-1:, :], far, axis=0)
    return np.vstack((l_pad, _arr, r_pad))


def _prepare_z_arr(
    img, mu=1/3, dpi=72, normalize=True, invert=False, pad=False, scale=1.0
):
    zar = np.array(_resize_img(img.copy().convert('L'), scale)).T
    zar = _invert_img_array(zar) if invert else zar
    zar = _normalize_img_array(zar, normalize)
    zar = _pad_img_array(zar, mu, dpi) if pad else zar
    return zar


def _color_palette_maker(palette='bw'):
    if palette in list(colormaps):
        color_palette = colormaps[palette](np.linspace(0, 1, 8))
        color_palette = np.round(color_palette[:, :3] * 255).astype('uint8')
    else:
        color_palette = np.array([(0, 0, 0), (255, 255, 255)], np.uint8)

    return color_palette


def _pattern_maker(size, ref, fit='fit', dpi=72, approach='rl'):
    w, h = size
    w_rep_len, h_rep_len = None, None
    repeat_len = int(round(dpi * 0.9))

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
            case 'mo' | 'random':
                lbound = round((asg.shape[1] - w) / 2)
                rbound = w + lbound
                asg = asg[:, lbound:rbound, lower:upper]
            case 'lr':
                asg = asg[:, :w, lower:upper]
            case _:     # rl as default
                asg = asg[:, -w:, lower:upper]

    return asg


def _sirds_init(zar, ref=None, fit='fit', dpi=72, approach='rl', palette='bw'):
    if ref is not None:
        asg = _pattern_maker(zar.shape, ref, fit, dpi, approach)
    else:
        color_pal = _color_palette_maker(palette)
        asg = np.array(
            color_pal[np.random.randint(len(color_pal), size=zar.shape)],
            dtype=np.uint8
        ).transpose(2, 0, 1)

    return asg


def _constrain(y, zar, mu=1/3, dpi=72, cross_eyed=False):
    w = zar.shape[0]
    eye_scalar = round(2.5 * dpi)
    same = list(range(w))

    for x in range(w):
        s = _pixel_separation(zar[x, y], mu, dpi, cross_eyed)
        left = x - round(s / 2)
        right = left + s

        if 0 <= left and right < w:
            t, zt, visible = _do_work(1, zar, x, y, mu, eye_scalar)
            while visible and (zt < 1):
                t, zt, visible = _do_work(t, zar, x, y, mu, eye_scalar)

            if visible:
                look = same[left]
                while look not in (left, right):
                    if look < right:
                        left = look
                        look = same[left]
                    else:
                        same[left] = right
                        left = right
                        look = same[left]
                        right = look
                same[left] = right

    return same


def _build_row(asg_row, constraints):
    for x in range(asg_row.shape[1]):
        if constraints[x] != x:
            if constraints[x] != -99:
                asg_row[:, x] = asg_row[:, constraints[x]]
            else:
                asg_row[:, x] = (255, 0, 0)

    return asg_row


def _sirds_row(asg, y, zar, mu=1/3, dpi=72, cross_eyed=False):
    asg_row = asg[:, :, y]
    return _build_row(asg_row, _constrain(y, zar, mu, dpi, cross_eyed))


class DisjointSet:
    """Data structure for traversing pixel constraints. Modified union find."""
    def __init__(self, list_size, approach='rl'):
        self.parent = list(range(list_size))
        self.approach = approach
        self.mp = (list_size - 1) / 2

    def find(self, idx):
        """Find the representative of a set."""
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.parent[idx])
        return self.parent[idx]

    def _prefer(self, idx, jdx):
        if self.approach == 'mo':
            if abs(idx - self.mp) < abs(jdx - self.mp):
                return idx, jdx
            return jdx, idx
        if self.approach == 'lr':
            return min(idx, jdx), max(idx, jdx)
        if self.approach == 'rl':
            return max(idx, jdx), min(idx, jdx)
        return np.random.choice([idx, jdx], size=2, replace=False).tolist()

    def unite(self, idx, jdx):
        """Join values."""
        irep, jrep = self.find(idx), self.find(jdx)
        if irep != jrep:
            root, other = self._prefer(irep, jrep)
            self.parent[other] = root


def _dsdsc(y, zar, mu=1/3, dpi=72, cross_eyed=False, approach='rl'):
    """Disjoint Set Data Structure Constrain"""
    w = zar.shape[0]
    eye_scalar = round(2.5 * dpi)
    constraints_structure = DisjointSet(w, approach)
    constraints = list(range(w))
    scan_order = constraints.copy()
    visible_pixels = [False] * w
    if approach == 'random':
        np.random.shuffle(scan_order)

    for x in scan_order:
        s = _pixel_separation(zar[x, y], mu, dpi, cross_eyed)
        left = x - round(s / 2)
        right = left + s

        if 0 <= left and right < w:
            t, zt, visible = _do_work(1, zar, x, y, mu, eye_scalar)
            while visible and (zt < 1):
                t, zt, visible = _do_work(t, zar, x, y, mu, eye_scalar)

            if visible:
                constraints_structure.unite(left, right)
                visible_pixels[left] = True
                visible_pixels[right] = True

    for i in constraints:
        if False:  # not visible_pixels[i]:
            constraints[i] = -99
        else:
            constraints[i] = constraints_structure.find(i)

    return constraints


def _asg_row(asg, y, zar, mu=1/3, dpi=72, cross_eyed=False, approach='rl'):
    asg_row = asg[:, :, y]
    return _build_row(asg_row, _dsdsc(y, zar, mu, dpi, cross_eyed, approach))


def _conv_dots(asg, depth, height='bottom', mu=1/3, dpi=72, cross_eyed=False):
    w, h = asg.shape[1:]
    s_pixels = asg.reshape(3, -1)
    colors, counts = np.unique(s_pixels, axis=1, return_counts=True)
    dot_color = colors[:, np.argmin(counts)]

    if 0 <= depth <= 1:
        how_far = _pixel_separation(depth, mu, dpi, cross_eyed)
        dot_r2 = (np.hypot(w, h) / 100) ** 2

        x_center_1 = w / 2 - how_far / 2
        x_center_2 = w / 2 + how_far / 2
        match height:
            case 'top':
                y_c_pos = 1
            case 'center':
                y_c_pos = 10
            case _:
                y_c_pos = 19
        y_center = h * y_c_pos / 20

        xs = np.arange(w)[:, None]
        ys = np.arange(h)[None, :]

        dot_1 = (xs - x_center_1) ** 2 + (ys - y_center) ** 2
        dot_2 = (xs - x_center_2) ** 2 + (ys - y_center) ** 2
        mask = (dot_1 < dot_r2) | (dot_2 < dot_r2)

        asg[:, mask] = dot_color[:, None]

    return asg
