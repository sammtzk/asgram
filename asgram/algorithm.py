# asgram/algorithm.py
"""
Constraint building algorithms for the creation of autostereograms.
"""

import numpy as np
try:
    from asgram.utils.tiw import _do_work
    from asgram.utils.utils import _pixel_separation
except ModuleNotFoundError:
    from utils.tiw import _do_work
    from utils.utils import _pixel_separation


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
            def _span_maker(mask):
                parents = output_arr.copy()
                parents[mask] = -1
                matches = parents == np.arange(len(parents))
                padded_matches = np.concat([[False], matches, [False]])
                deltas = (np.where(np.diff(padded_matches))[0]).tolist()
                assert len(deltas[0::2]) == len(deltas[1::2])
                return [(s, e) for s, e in zip(deltas[0::2], deltas[1::2])]

            uncon_mask = ~(np.asarray(self.constrained))
            uncon_spans = _span_maker(uncon_mask)
            uncon_spans = [(span, True) for span in uncon_spans]
            root_spans = _span_maker(~uncon_mask)
            root_spans = [(span, False) for span in root_spans]
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
    return pixel_map.constraints


def _build_row_vectorized(asg_row, constraints):
    constraints = np.asarray(constraints)
    pointers = (constraints != np.arange(constraints.size))
    asg_row[:, pointers] = asg_row[:, constraints[pointers]]
    return asg_row


def _asg_row(
    asg, y, zar, _re=False, mu=1/3, dpi=72, cross_eyed=False, approach='rl'
):
    asg_row = asg[:, :, y]
    constraints = _dsdsc(y, zar, _re, mu, dpi, cross_eyed, approach)
    return _build_row_vectorized(asg_row, constraints)
