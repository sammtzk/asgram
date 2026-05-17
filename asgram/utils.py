# sgram/utils.py
"""
Utility functions for the creation of autostereograms. I should make an asg
params data class.
"""

import numpy as np
import cv2 as cv
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


def _kitsch_smooth_img_array(_arr, mu=1/3, dpi=72, smoothing_amt=0):
    if smoothing_amt > 0:
        far = _pixel_separation(0, mu, dpi, cross_eyed=False)
        near = _pixel_separation(1, mu, dpi, cross_eyed=False)
        detail = np.ceil(255 / (far - near))

        ls_arr = np.roll(_arr, shift=-1, axis=0)
        rs_arr = np.roll(_arr, shift=1, axis=0)

        rmask = np.abs(_arr - rs_arr) < detail
        lmask = np.abs(_arr - ls_arr) < detail

        mask = rmask | lmask
        ls_amt = -int(round(smoothing_amt / 2))
        rs_amt = ls_amt + smoothing_amt
        for i in range(abs(ls_amt)):
            mask = mask & np.roll(mask, shift=-(i + 1), axis=0)
        for i in range(abs(rs_amt)):
            mask = mask & np.roll(mask, shift=(i + 1), axis=0)

        # pylint: disable=no-member
        hsmooth = cv.GaussianBlur(_arr, (1, smoothing_amt), 0)
        _arr[mask] = hsmooth[mask]

    return _arr


def _smooth_img_array(_arr):
    # pylint: disable=no-member
    return cv.bilateralFilter(_arr, 9, 75, 75)


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
    img, mu=1/3, dpi=72,
    normalize=True, invert=False, smooth=False, pad=False, scale=1.0
):
    zar = np.array(_resize_img(img.copy().convert('L'), scale)).T
    zar = _invert_img_array(zar) if invert else zar
    zar = _normalize_img_array(zar, normalize)
    zar = _smooth_img_array(zar) if smooth else zar
    zar = _pad_img_array(zar, mu, dpi) if pad else zar
    return zar


def _color_palette_maker(palette='bw'):
    if palette in list(colormaps):
        color_palette = colormaps[palette](np.linspace(0, 1, 8))
        color_palette = np.round(color_palette[:, :3] * 255).astype('uint8')
    else:
        color_palette = np.array([(0, 0, 0), (255, 255, 255)], np.uint8)

    return color_palette


def _pattern_maker(size, ref, fit='fit', mu=1/3, dpi=72, approach='rl'):
    w, h = size
    w_rep_len, h_rep_len = None, None
    repeat_len = _pixel_separation(0, mu, dpi, cross_eyed=False)

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
    zar, ref=None, fit='fit', mu=1/3, dpi=72, approach='rl', palette='bw'
):
    if ref is not None:
        asg = _pattern_maker(zar.shape, ref, fit, mu, dpi, approach)
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
            asg_row[:, x] = asg_row[:, constraints[x]]

    return asg_row


def _sirds_row(asg, y, zar, mu=1/3, dpi=72, cross_eyed=False):
    asg_row = asg[:, :, y]
    constraints = _constrain(y, zar, mu, dpi, cross_eyed)
    return _build_row(asg_row, constraints)


class DisjointSet:
    """Data structure for traversing pixel constraints. Modified union find."""
    def __init__(self, list_size, mu=1/3, dpi=72, approach='rl'):
        self.size = list_size
        self.far = _pixel_separation(0, mu, dpi, cross_eyed=False)
        self.parent = list(range(self.size))
        self.depths = [0.0] * self.size
        self.approach = approach
        self.mp = (self.size - 1) / 2

        if self.approach in ['rl', 'lr', 'mo']:
            if self.approach == 'rl':
                ma = self.size
                mi = ma - self.far
            elif self.approach == 'lr':
                mi = 0
                ma = mi + self.far
            else:   # 'mo'
                mi = round(self.mp - self.far / 2)
                ma = mi + self.far

            self.src = np.linspace(mi, ma - 1, self.far).astype(int).tolist()
            assert len(self.src) == self.far

        elif self.approach == 'oi':
            lsize = round(self.far / 2)
            lmi = 0
            lma = lmi + lsize
            lsource = np.linspace(lmi, lma - 1, num=lsize).astype(int).tolist()

            rsize = self.far - lsize
            rma = self.size
            rmi = rma - rsize
            rsource = np.linspace(rmi, rma - 1, num=rsize).astype(int).tolist()

            self.src = lsource + rsource
            assert len(self.src) == self.far

        else:
            self.src = None

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
        if self.approach == 'oi':
            if abs(idx - self.mp) > abs(jdx - self.mp):
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

    def depth_unite(self, idx, jdx, source_depth):
        """Join values."""
        irep, jrep = self.find(idx), self.find(jdx)
        self.depths[irep] = max(self.depths[irep], source_depth)
        self.depths[jrep] = max(self.depths[jrep], source_depth)

        if irep != jrep:
            irep_depth, jrep_depth = self.depths[irep], self.depths[jrep]
            if irep_depth > jrep_depth:
                self.parent[jrep] = irep
            elif irep_depth < jrep_depth:
                self.parent[irep] = jrep
            else:
                root, other = self._prefer(irep, jrep)
                self.parent[other] = root

    def unite_to_neighbor(self, idx, return_new_root=False):
        """Join a value to a neighboring root."""
        left = idx - 1
        right = idx + 1
        if 0 <= left:
            if self.size > right:
                new_root, _ = self._prefer(self.find(left), self.find(right))
            else:
                new_root = self.find(left)
        else:
            new_root = self.find(right)

        self.parent[idx] = new_root

        if return_new_root:
            return new_root

    def _reassign_root(self, old_root, new_root):
        for i in range(self.size):
            if self.parent[i] == old_root:
                self.parent[i] = new_root

    def enforce_source(self, idx):
        """Ensure that values have roots from source."""
        if self.src is not None:
            old_root = self.parent[idx]
            if old_root not in self.src:
                new_root = self.unite_to_neighbor(idx, return_new_root=True)
                self._reassign_root(old_root, new_root)

    def global_enforce_source(self):
        """Ensure that all values have roots from source."""
        if self.src is not None:
            for i in range(self.size):
                self.enforce_source(i)

    def enforce_nearby(self, idx, thresh=9):
        """Ensure that a pixel draws from similar nearby sources."""
        old_root = self.parent[idx]
        switch_flag = 0

        lrep = self.find(idx - 1) if idx > 0 else np.nan
        if not np.isnan(lrep):
            if abs(lrep - old_root) > thresh:
                switch_flag += 1
        else:
            switch_flag += 1

        rrep = self.find(idx + 1) if idx < self.size - 1 else np.nan
        if not np.isnan(rrep):
            if abs(rrep - old_root) > thresh:
                switch_flag += 1
        else:
            switch_flag += 1

        if switch_flag == 2:
            if np.isnan(lrep) or np.isnan(rrep):
                _mp01 = np.nanmean([lrep, rrep])   # type: ignore
                nr0 = np.floor(_mp01).astype(int)
                nr1 = np.ceil(_mp01).astype(int)
                new_root, _ = self._prefer(nr0, nr1)
            else:
                nr, _ = self._prefer(lrep, rrep)
                sf = 0

                nr0 = self.find(idx - 1) if idx > 0 else np.nan
                if not np.isnan(nr0):
                    if abs(nr0 - old_root) > thresh:
                        sf += 1
                else:
                    sf += 1

                nr1 = self.find(idx + 1) if idx < self.size - 1 else np.nan
                if not np.isnan(nr1):
                    if abs(nr1 - old_root) > thresh:
                        sf += 1
                else:
                    sf += 1

                if sf == 2:
                    if np.isnan(nr0) or np.isnan(nr1):
                        _mp01 = np.nanmean([nr0, nr1])   # type: ignore
                        nr2 = np.floor(_mp01).astype(int)
                        nr3 = np.ceil(_mp01).astype(int)
                        new_root, _ = self._prefer(nr2, nr3)
                    else:
                        new_root, _ = self._prefer(nr0, nr1)
                else:
                    new_root = nr

            self._reassign_root(old_root, new_root)

    def global_enforce_nearby(self, thresh=9):
        """Ensure unconstrained pixels draw from similar nearby sources."""
        for i in range(self.size):
            old_root = self.parent[i]
            if old_root == i:
                switch_flag = 0

                lrep = self.find(i - 1) if i > 0 else np.nan
                if not np.isnan(lrep):
                    if abs(lrep - old_root) > thresh:
                        switch_flag += 1
                else:
                    switch_flag += 1

                rrep = self.find(i + 1) if i < self.size - 1 else np.nan
                if not np.isnan(rrep):
                    if abs(rrep - old_root) > thresh:
                        switch_flag += 1
                else:
                    switch_flag += 1

                if switch_flag == 2:
                    if np.isnan(lrep) or np.isnan(rrep):
                        _mp01 = np.nanmean([lrep, rrep])   # type: ignore
                        nr0 = np.floor(_mp01).astype(int)
                        nr1 = np.ceil(_mp01).astype(int)
                        new_root, _ = self._prefer(nr0, nr1)
                    else:
                        new_root, _ = self._prefer(lrep, rrep)
                    self._reassign_root(old_root, new_root)


def _dsdsc(y, zar, ref=None, mu=1/3, dpi=72, cross_eyed=False, approach='rl'):
    """Disjoint Set Data Structure Constrain"""
    w = zar.shape[0]
    eye_scalar = round(2.5 * dpi)
    constraints_structure = DisjointSet(w, mu, dpi, approach)
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
                # constraints_structure.depth_unite(left, right, zar[x, y])
                visible_pixels[left] = True
                visible_pixels[right] = True

    # constraints_structure.global_enforce_nearby(9)

    for i in constraints:
        if not visible_pixels[i] and ref is not None:
            constraints_structure.enforce_nearby(i)
        constraints[i] = constraints_structure.find(i)

    return constraints


def _build_row_vectorized(asg_row, constraints):
    constraints = np.asarray(constraints)
    same = constraints == np.arange(constraints.size)
    not_visible = constraints == -99
    pointers = ~(same | not_visible)

    if np.any(pointers):
        asg_row[:, pointers] = asg_row[:, constraints[pointers]]

    if np.any(not_visible):
        asg_row[:, not_visible] = np.array([255, 0, 0])[:, None]

    return asg_row


def _redmean_color_diff(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    rmean = 0.5 * (r1 + r2)
    drs = (r1 - r2) ** 2
    dgs = (g1 - g2) ** 2
    dbs = (b1 - b2) ** 2

    rw = 2 + rmean / 256
    gw = 4
    bw = 2 + (255 - rmean) / 256

    return np.sqrt(rw * drs + gw * dgs + bw * dbs)


def _pdvrpp(asg_row):
    """Pixel Disparity Visual Rectification Post-Processing"""

    corrected_asg_row = asg_row.copy()
    for inner_idx in np.arange(0 + 1, asg_row.shape[1] - 1):
        target_pixel = asg_row[:, inner_idx]
        left_pixel = asg_row[:, inner_idx + 1]
        right_pixel = asg_row[:, inner_idx - 1]

        left_rmcd = _redmean_color_diff(target_pixel, left_pixel)
        right_rmcd = _redmean_color_diff(target_pixel, right_pixel)

        if 100 < left_rmcd and right_rmcd < 100:
            rbar = abs(left_pixel[0] - right_pixel[0])
            gbar = abs(left_pixel[0] - right_pixel[0])
            bbar = abs(left_pixel[0] - right_pixel[0])
            corrected_asg_row[:, target_pixel] = [rbar, gbar, bbar]

    return corrected_asg_row


def _asg_row(
    asg, y, zar, ref=None, mu=1/3, dpi=72, cross_eyed=False, approach='rl'
):
    asg_row = asg[:, :, y]
    constraints = _dsdsc(y, zar, ref, mu, dpi, cross_eyed, approach)
    asg_row = _build_row_vectorized(asg_row, constraints)
    return _pdvrpp(asg_row)


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
