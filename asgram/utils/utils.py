# asgram/utils/utils.py
"""
Utility methods for the asgram package.
"""

try:
    from asgram.utils.tiw import _separation
except ModuleNotFoundError:
    from utils.tiw import _separation


def _pixel_separation(Z, mu=1/3, dpi=72, cross_eyed=False):
    """Modifies the TIW approach to allow for cross-view autostereograms."""
    Z = -Z + 1.0 if cross_eyed else Z
    return _separation(Z=Z, mu=mu, dpi=dpi)
