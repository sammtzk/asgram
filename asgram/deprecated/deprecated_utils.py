# asgram/utils.py
"""
Utility functions for the creation of autostereograms.
"""

try:
    from asgram.tiw import _do_work, _separation
except ModuleNotFoundError:
    from tiw import _do_work, _separation


def _pixel_separation(Z, mu=1/3, dpi=72, cross_eyed=False):
    Z = -Z + 1.0 if cross_eyed else Z
    return _separation(Z=Z, mu=mu, dpi=dpi)


def _constrain(y, zar, mu=1/3, dpi=72, cross_eyed=False):
    """Deprecated."""
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
    """Deprecated."""
    for x in range(asg_row.shape[1]):
        if constraints[x] != x:
            asg_row[:, x] = asg_row[:, constraints[x]]

    return asg_row


def _sirds_row(asg, y, zar, mu=1/3, dpi=72, cross_eyed=False):
    """Deprecated"""
    asg_row = asg[:, :, y]
    constraints = _constrain(y, zar, mu, dpi, cross_eyed)
    return _build_row(asg_row, constraints)
