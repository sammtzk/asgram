# asgram/algorithm.py
"""
Utility functions for the creation of autostereograms. I should make an asg
params data class.
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


def _dsdsc(y, zar, _re=False, mu=1/3, dpi=72, cross_eyed=False, approach='rl'):
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
        if not visible_pixels[i] and _re:
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


def _asg_row(
    asg, y, zar, _re=False, mu=1/3, dpi=72, cross_eyed=False, approach='rl'
):
    asg_row = asg[:, :, y]
    constraints = _dsdsc(y, zar, _re, mu, dpi, cross_eyed, approach)
    return _build_row_vectorized(asg_row, constraints)
