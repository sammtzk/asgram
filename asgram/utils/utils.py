# asgram/utils/utils.py
"""
Utility methods for the asgram package.
"""

from dataclasses import dataclass
from matplotlib import colormaps
try:
    from asgram.utils.tiw import _separation
except ModuleNotFoundError:
    from utils.tiw import _separation


def _pixel_separation(Z, mu=1/3, dpi=72, cross_eyed=False):
    """Modifies the TIW approach to allow for cross-view autostereograms."""
    Z = -Z + 1.0 if cross_eyed else Z
    return _separation(Z=Z, mu=mu, dpi=dpi)


@dataclass
class AllParameters():
    """Mutable runtime variables for asgram generation."""
    # fundamental
    depth_of_field: float = 1/3                 # mu
    dots_per_inch: int = 72                     # dpi
    cross_view_flag: bool = True                # cross [asgram default=False]
    constraint_approach: str = 'rl'             # approach
    parallelization_cores: int = 16             # num_jobs [asgram default=8]

    # depth map
    normalize_depth_map: bool = True            # normalize
    invert_depth_map: bool = False              # invert
    depth_map_smoothing: bool = False           # iis
    depth_map_bilateral_filter: bool = False    # bil
    pad_depth_map: bool = True                  # pad [asgram default=False]
    scale_depth_map: float = 1.0                # scale

    # image pattern
    pattern_fit: str = 'fit'                    # rfit
    random_pattern_palette: str = 'bw'          # rpal
    random_seed: int = 1132                     # random_seed

    # postprocessing
    pixel_disparity_smoothing: bool = False     # pdvrs
    convergence_dot_depth: float = -1.0         # dot_depth [asgram default=0.0]  # noqa E501
    convergence_dot_placement: str = 'top'      # dot_height [asgram default='bottom']  # noqa E501

    def item_lookup(self, field_name):
        if hasattr(self, field_name):
            match field_name:
                case 'constraint_approach':
                    return {
                        'Right to Left': 'rl',
                        'Left to Right': 'lr',
                        'Middle Outward': 'mo',
                        'Outer Inward': 'oi',
                        'True Random': 'random'
                    }
                case 'pattern_fit':
                    return {
                        'Fit': 'fit',
                        'Auto Tile': 'tile',
                        'Horizontal Tile': 'htile',
                        'Vertical Tile': 'vtile',
                        'Enforce Approach-Specific Source': 'ES'
                    }
                case 'random_pattern_palette':
                    palettes = ['Black and White']
                    palettes.extend(list(colormaps))
                    pal_dict = {pal: pal for pal in palettes}
                    pal_dict['Black and White'] = 'bw'
                    return pal_dict
                case 'convergence_dot_depth':
                    return {
                        'None': -1,
                        'Far Plane': 0,
                        'Near Plane': 1
                    }
                case 'convergence_dot_placement':
                    return {
                        'Top': 'top',
                        'Center': 'center',
                        'Bottom': 'bottom'
                    }
        return {}
