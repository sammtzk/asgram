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
    from asgram.source_pattern_making import SrcPat
    from asgram.utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )
except ModuleNotFoundError:
    from utils.tiw import _do_work
    from utils.utils import _pixel_separation
    from depth_map_making import _normalize_img_array, ZMap
    from source_pattern_making import SrcPat
    from utils.parallelize import (
        worker_count, run_worker, parallelize_workers
    )


class DisjointSet:
    """Data structure for traversing pixel constraints. Modified union find."""
    def __init__(self, list_size, mu=1/3, dpi=72, approach='rl'):
        self.size = list_size
        self.far = _pixel_separation(0, mu, dpi, cross_eyed=False)
        self.parent = list(range(self.size))
        self.constrained = [False] * self.size
        # self.recursion_safety = 0
        self.approach = approach
        self.mp = (self.size - 1) / 2
        self.src = self._source_specification()
        if self.src is not None:
            assert len(self.src) == self.far

    def _source_specification(self):
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

            return np.arange(mi, ma)

        elif self.approach == 'oi':
            lsize = round(self.far / 2)
            lmi = 0
            lma = lmi + lsize
            lsource = np.arange(lmi, lma)

            rsize = self.far - lsize
            rma = self.size
            rmi = rma - rsize
            rsource = np.arange(rmi, rma)

            return np.concat([lsource, rsource])

        else:
            return None

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

    def _boundary_prefer(self, l_idx, r_idx, zar_row=None):
        assert not ((l_idx is None) and (r_idx is None))
        if l_idx is None:
            return self.find(r_idx), 'r'
        elif r_idx is None:
            return self.find(l_idx), 'l'
        else:
            lrep, rrep = self.find(l_idx), self.find(r_idx)
            if zar_row is not None:
                lz, rz = zar_row[lrep], zar_row[rrep]
                root = lrep if lz < rz else rrep
            else:
                root, _ = self._prefer(lrep, rrep)
            return (root, 'l') if root == lrep else (root, 'r')

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
        self.constrained[idx] = True
        self.constrained[jdx] = True

        irep, jrep = self.find(idx), self.find(jdx)
        if irep != jrep:
            root, other = self._prefer(irep, jrep)
            self.parent[other] = root

    @property
    def constraints(self):
        """Return the parent for each index."""
        return [self.find(i) for i in range(self.size)]

    def shift_oos_roots(self, zar_row=None):
        """
        Identify out of source roots, including unconstrained pixels, and shift
        them to values within the source range.
        """
        output_arr = np.array(self.constraints)     # type: ignore
        if self.src is not None:
            def _span_maker(mask, use_z):
                parents = output_arr.copy()
                parents[mask] = -1
                matches = parents == np.arange(len(parents))
                padded_matches = np.concat([[False], matches, [False]])
                deltas = (np.where(np.diff(padded_matches))[0]).tolist()
                assert len(deltas[0::2]) == len(deltas[1::2])
                spans = [(s, e) for s, e in zip(deltas[0::2], deltas[1::2])]
                return [(span, use_z) for span in spans]

            uncon_mask = ~(np.asarray(self.constrained))
            uncon_spans = _span_maker(uncon_mask, use_z=True)
            root_spans = _span_maker(~uncon_mask, use_z=False)
            all_spans = uncon_spans + root_spans

            # establish bounds for approach-specific sources
            src_l, src_u = min(self.src).item(), max(self.src).item()
            for span, use_z in all_spans:
                # check whether pixels in span partially originate from source
                s_idx, e_idx = span
                if (s_idx in self.src) or (e_idx in self.src):
                    continue

                # if oos, unite to either left pixels or right pixels
                l_idx = s_idx - 1 if 0 < s_idx else None
                r_idx = e_idx + 1 if self.size - 1 > e_idx else None
                zar_row_val = zar_row if use_z else None
                anchor, side = self._boundary_prefer(l_idx, r_idx, zar_row_val)

                # calculate shift based on anchor point
                reference = s_idx if 'l' == side else e_idx - 1
                shift_oos = anchor - reference

                # apply shift to anchor for values in span
                for idx in range(s_idx, e_idx):
                    # check bounds
                    new_parent = idx + shift_oos
                    if self.size <= new_parent:
                        in_bounds = False
                        while not in_bounds:
                            new_parent -= self.far
                            if (src_l <= new_parent) and (src_u >= new_parent):
                                in_bounds = True
                        shift = new_parent - idx
                    elif 0 > new_parent:
                        in_bounds = False
                        while not in_bounds:
                            new_parent += self.far
                            if (src_l <= new_parent) and (src_u >= new_parent):
                                in_bounds = True
                        shift = new_parent - idx
                    else:
                        shift = shift_oos

                    # shift individual pixel and reassign the root
                    output_arr[idx] += shift
                    if not use_z:
                        output_arr[output_arr == idx] = output_arr[idx]
        self.parent = output_arr.tolist()


def _dsdsc(y, zar, _re=False, mu=1/3, dpi=72, cross_eyed=False, approach='rl'):
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
                pixel_map.unite(left, right)

    if _re:
        pixel_map.shift_oos_roots(zar[:, y])
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
            approach=ad['approach']
        )

    return run_worker(ys_to_build, args_dict, _row_func_wrapper, 2, np.uint16)


class PixCon:
    """Calculates and stores pixel constraints for autostereograms."""

    def __init__(
            self, zmap: ZMap, sp: SrcPat,
            mu=1/3, dpi=72, cross=False, approach='rl', num_jobs=8
    ):
        self.zmap = zmap
        self.sp = sp

        self.mu = mu
        self.dpi = dpi
        self.cross = cross
        self.approach = approach
        self.num_jobs = num_jobs

        self.con_mat = np.array([])
        self.con_img = Image.new('1', (0, 0))
        self.asg_mat = np.array([])
        self.asg_img = Image.new('1', (0, 0))
        self.update()

    @property
    def zar(self):
        return self.zmap.zm_arr

    @property
    def total_ys(self):
        return self.zmap.zm_arr.shape[1]

    @property
    def _sp(self):
        return self.sp.sp_arr

    @property
    def _re(self):
        return self.sp.ref is not None

    @property
    def args_dict(self):
        return {
            'src_mat': self.zar,
            'total_ys': self.total_ys,
            '_re': self._re,
            'mu': self.mu,
            'dpi': self.dpi,
            'cross': self.cross,
            'approach': self.approach
        }

    def _img_updates(self, pc):
        self.con_mat = pc
        self.con_img = Image.fromarray(
            _normalize_img_array(pc).T * 255.0
        ).convert('L')
        self.asg_mat = np.take_along_axis(self._sp, pc[None, :, :], axis=1)
        self.asg_img = Image.fromarray(self.asg_mat.T).convert('RGB')

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
                    self.mu, self.dpi, self.cross, self.approach
                )

        self._img_updates(con)
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
            'num_jobs': self.num_jobs,
            'pixcon_path': path1
        }

        with open(path1, 'wb') as f:
            np.save(f, self.con_mat, allow_pickle=True)
        with open(path2, 'w') as f:
            json.dump(params_dict, f)
