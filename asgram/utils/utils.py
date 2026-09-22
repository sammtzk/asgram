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


def pixel_separation_summary(mu=1/3, dpi=72, cross_eyed=False):
    """Summary of pixel separation behavior based on input parameters."""
    far = _pixel_separation(0, mu, dpi, cross_eyed)
    near = _pixel_separation(1, mu, dpi, cross_eyed)

    print(
        f"Separation at the Far plane:  {far} px\n"
        f"Separation at the Near plane: {near} px\n"
        f"Number of possible values:    {abs(near - far) + 1}"
    )
