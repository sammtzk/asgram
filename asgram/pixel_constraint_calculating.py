# asgram/pixel_constraint_calculating.py
"""
Constraint building algorithms for the creation of autostereograms.
"""

import json
import numpy as np
from PIL import Image
try:
    from asgram.utils.tiw import _do_work
    from asgram.utils.utils import _pixel_separation
    from asgram.depth_map_making import _normalize_img_array, ZMap
    from asgram.utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )
except ModuleNotFoundError:
    from utils.tiw import _do_work
    from utils.utils import _pixel_separation
    from depth_map_making import _normalize_img_array, ZMap
    from utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )


class DisjointSet:
    """Data structure for traversing pixel constraints. Modified union find."""

    def __init__(self, list_size, mu=1/3, dpi=72, approach='rl'):
        self.size = list_size
        self.far = _pixel_separation(0, mu, dpi, cross_eyed=False)
        self.parent = list(range(self.size))
        self.approach = approach
        self.mp = (self.size - 1) / 2

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

    def find(self, idx):
        """Find the representative of a set."""
        try:
            if self.parent[idx] != idx:
                self.parent[idx] = self.find(self.parent[idx])
        except RecursionError:
            while self.parent[idx] != idx:
                self.parent[idx] = self.parent[self.parent[idx]]
                idx = self.parent[idx]
        return self.parent[idx]

    def unite(self, idx, jdx):
        """Join values."""
        irep, jrep = self.find(idx), self.find(jdx)
        if irep != jrep:
            root, other = self._prefer(irep, jrep)
            self.parent[other] = root

    @property
    def constraints(self):
        """Return the parent for each index."""
        return [self.find(i) for i in range(self.size)]

    def shift_oos_roots(self, src_area_row=None):
        """
        Identify out of source roots, including unconstrained pixels, and shift
        them to values within the source range.
        """
        if src_area_row is not None:
            output_arr = np.array(self.constraints)  # type: ignore

            def _oos_spans_finder():
                parents = output_arr.copy()
                parents[src_area_row] = -1
                matches = parents == np.arange(len(parents))
                padded_matches = np.concat([[False], matches, [False]])
                deltas = (np.where(np.diff(padded_matches))[0]).tolist()
                assert len(deltas[0::2]) == len(deltas[1::2])
                return [(s, e) for s, e in zip(deltas[0::2], deltas[1::2])]

            src_idxs = np.where(src_area_row)[0]
            max_sep = len(src_idxs)

            def _shift_spans(_spans):
                for s_idx, e_idx in _spans:
                    span_len = e_idx - s_idx
                    _l = s_idx - 1 if 0 < s_idx else None
                    _r = e_idx + 1 if self.size - 1 > e_idx else None

                    do_interpolation = False
                    if ((span_len > 1) and (None not in [_l, _r])):
                        if (all(self.find(_i) in src_idxs for _i in [_l, _r])):
                            do_interpolation = True

                    if do_interpolation:
                        # robust index interpolation for oi source areas
                        l_idx = np.argmax(self.find(_l) == src_idxs).item()
                        r_idx = np.argmax(self.find(_r) == src_idxs).item()

                        if l_idx < r_idx:
                            _in_idxs = np.linspace(l_idx, r_idx, span_len + 2)
                        else:
                            _in_idxs = np.concat([
                                np.arange(l_idx, max_sep),
                                np.arange(0, r_idx + 1)
                            ])
                            _in_idxs = _in_idxs[
                                np.round(np.linspace(
                                    0, len(_in_idxs) - 1, span_len + 2
                                )).astype(np.uint8)
                            ]

                        _in_idxs = np.round(_in_idxs).astype(np.uint16)
                        insert = src_idxs[_in_idxs]
                    else:
                        assert (_l is not None) or (_r is not None)
                        if _l is None:
                            anchor, side = self.find(_r), 'r'
                        elif _r is None:
                            anchor, side = self.find(_l), 'l'
                        else:
                            lrep, rrep = self.find(_l), self.find(_r)
                            anchor, _ = self._prefer(lrep, rrep)
                            side = 'l' if anchor == lrep else 'r'
                        anchor_idx = np.argmax(anchor == src_idxs).item()

                        if 'l' == side:
                            insert = np.array([
                                src_idxs[(anchor_idx + i) % max_sep]
                                for i in range(span_len + 2)
                            ])
                        else:  # 'r' == side
                            insert = np.array([
                                src_idxs[(anchor_idx - i) % max_sep]
                                for i in range(span_len + 2)
                            ][::-1])

                    output_arr[s_idx:e_idx] = insert[1:-1]  # remove buffer
                self.parent = output_arr.tolist()

            _shift_spans(_oos_spans_finder())


def _dsdsc(
        y, zar, _re=False,
        mu=1/3, dpi=72, cross_eyed=False, approach='rl',
        src_area=None
):
    """Disjoint Set Data Structure Constrain"""
    w = zar.shape[0]
    eye_scalar = round(2.5 * dpi)
    pixel_map = DisjointSet(w, mu, dpi, approach)
    scan_order = list(range(w))
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
                pixel_map.unite(left, right)  # keep l then r

    if _re and (src_area is not None):
        pixel_map.shift_oos_roots(src_area_row=src_area[:, y])
    return np.asarray(pixel_map.constraints, dtype=np.uint16)


def _pixcon_worker(_args):
    """Worker for PixCon parallelization. Wraps generic run_worker."""
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
            approach=ad['approach'],
            src_area=ad['src_area']
        )

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, 2, np.uint16)


class PixCon:
    """Calculates and stores pixel constraints for autostereograms."""

    def __init__(
            self, zmap: ZMap,
            mu=1/3, dpi=72, cross=False, approach='rl', fill=False, num_jobs=8
    ):
        self.zmap = zmap

        self.mu = mu
        self.dpi = dpi
        self.cross = cross
        self.approach = approach
        self.fill = fill
        self.num_jobs = num_jobs

        self.con_mat = np.array([])
        self.con_img = Image.new('1', (0, 0))
        self.update()

    @property
    def zar(self):
        return self.zmap.zm_arr

    @property
    def total_ys(self):
        return self.zmap.zm_arr.shape[1]

    @property
    def _re(self):
        return self.fill

    @property
    def src_area(self):
        """
        Uses depth map information to determine the size and shape of the
        source area for pixels fit to a constraint matrix. The source area can
        be used to determine which pixels should be roots -- pointers which are
        their own parent -- thereby replacing the earlier source specification
        method in DisjointSet. Can also be used for source pattern construction
        for smooth image tiling.

        Returns source area as a bool array where True indicates source pixel.
        """
        if self.approach not in ['mo', 'oi', 'lr', 'rl']:
            return None
        zm_arr = self.zmap.zm_arr
        src_area = np.zeros_like(zm_arr, dtype=bool)
        width, height = self.zmap.size

        for y in range(height):
            _z = np.max(zm_arr[:, y]) if self.cross else np.min(zm_arr[:, y])
            max_sep = _pixel_separation(_z, self.mu, self.dpi, self.cross)

            if 'oi' != self.approach:
                match self.approach:
                    case 'mo':
                        row_min = int(np.ceil((width - max_sep) / 2))
                    case 'lr':
                        row_min = 0
                    case _:  # rl
                        row_min = width - max_sep

                row_max = row_min + max_sep
                src_area[row_min:row_max, y] = True

            else:
                lsize = int(np.floor(max_sep / 2))
                rsize = max_sep - lsize
                src_area[0:lsize, y] = True
                src_area[(width - rsize):width, y] = True

            assert max_sep == np.sum(src_area[:, y])

        return src_area

    @property
    def args_dict(self):
        return {
            'src_mat': self.zar,
            'total_ys': self.total_ys,
            '_re': self._re,
            'mu': self.mu,
            'dpi': self.dpi,
            'cross': self.cross,
            'approach': self.approach,
            'src_area': self.src_area
        }

    def update(self):
        """Updates the pixel constraints according to class parameters."""
        print("Step: Pixel Constraint Calculating")
        jobs = worker_count(self.num_jobs)
        if 1 < jobs:
            con = parallelize_workers(
                self.args_dict, _pixcon_worker, jobs, np.uint16
            )
        else:
            con = np.zeros_like(self.zar, dtype=np.uint16)
            for y in range(self.total_ys):
                con[:, y] = _dsdsc(
                    y, self.zar, self._re,
                    self.mu, self.dpi, self.cross, self.approach, self.src_area
                )

        self.con_mat = con
        self.con_img = Image.fromarray(
            _normalize_img_array(con).T * 255.0
        ).convert('L')
        print("Complete.")

    def save(self, file_name='temp', dir_path=''):
        """
        Saves pixel constraint matrix and fundamental parameters.

        Saves two files:
            1. a .npy format Python pickle of self.con_mat
            2. a .json of fundamental parameters with a file path to 1.

        Both files must exist in order to synthesize new asgrams down the line.
        """
        dir_path = dir_path + '/' if dir_path else dir_path
        file_path = dir_path + file_name
        path1 = file_path + '.npy'
        path2 = file_path + '.json'

        params_dict = {
            'mu': self.mu,
            'dpi': self.dpi,
            'cross': self.cross,
            'approach': self.approach,
            'fill': self.fill,
            'num_jobs': self.num_jobs,
            'pixcon_path': path1
        }

        with open(path1, 'wb') as f:
            np.save(f, self.con_mat, allow_pickle=True)
        with open(path2, 'w') as f:
            json.dump(params_dict, f)
