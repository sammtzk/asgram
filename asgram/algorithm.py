# asgram/algorithm.py
"""
Utility functions for the creation of autostereograms.
"""

import numpy as np
try:
    from asgram.tiw import _do_work, _separation
except ModuleNotFoundError:
    from tiw import _do_work, _separation


def _pixel_separation(Z, mu=1/3, dpi=72, cross_eyed=False):
    Z = -Z + 1.0 if cross_eyed else Z
    return _separation(Z=Z, mu=mu, dpi=dpi)


class DisjointSet:
    """Data structure for traversing pixel constraints. Modified union find."""
    def __init__(self, list_size, mu=1/3, dpi=72, approach='rl'):
        self.size = list_size
        self.far = _pixel_separation(0, mu, dpi, cross_eyed=False)
        self.parent = list(range(self.size))
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

    def find(self, idx):
        """Find the representative of a set."""
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.parent[idx])
        return self.parent[idx]

    def unite(self, idx, jdx):
        """Join values."""
        irep, jrep = self.find(idx), self.find(jdx)
        if irep != jrep:
            root, other = self._prefer(irep, jrep)
            self.parent[other] = root

    def _boundary_prefer(self, l_idx, r_idx):
        assert not ((l_idx is None) and (r_idx is None))
        if l_idx is None:
            return self.find(r_idx), 'r'
        elif r_idx is None:
            return self.find(l_idx), 'l'
        else:
            lrep, rrep = self.find(l_idx), self.find(r_idx)
            root, _ = self._prefer(lrep, rrep)
            return (root, 'l') if root == lrep else (root, 'r')

    def shift_oos_roots(self):
        """
        Identify out of source roots, including unconstrained pixels, and shift
        them to values within the source range.
        """
        output_arr = np.array([self.find(i) for i in range(self.size)])
        if self.src is not None:
            parents = output_arr.copy()
            poss = np.arange(len(parents))
            padded_bool_arr = np.concat([[False], parents == poss, [False]])
            deltas = (np.where(np.diff(padded_bool_arr))[0]).tolist()
            assert len(deltas[0::2]) == len(deltas[1::2])
            spans = [(s, e) for s, e in zip(deltas[0::2], deltas[1::2])]

            # establish bounds for approach-specific sources
            src_l = min(self.src).item()
            src_u = max(self.src).item()

            for s_idx, e_idx in spans:
                # check whether pixels in span partially originate from source
                if (s_idx in self.src) or (e_idx in self.src):
                    continue

                # if oos, unite to either left pixels or right pixels
                l_idx = s_idx - 1 if 0 < s_idx else None
                r_idx = e_idx + 1 if self.size - 1 > e_idx else None
                anchor, direction = self._boundary_prefer(l_idx, r_idx)

                # calculate shift based on anchor point
                reference = s_idx if 'l' == direction else e_idx - 1
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

                    # shift pixel by pixel
                    output_arr[idx] += shift
        self.parent = output_arr.tolist()

    @property
    def constraints(self):
        """Return the parent for each index."""
        return [self.find(i) for i in range(self.size)]


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
        pixel_map.shift_oos_roots()
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
