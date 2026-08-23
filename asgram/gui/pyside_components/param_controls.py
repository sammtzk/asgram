# asgram/gui/pyside_components/param_controls.py
"""
The parameter control panel for the asgram PySide6 app. Establishes the
framework for globally shared parameters and the layout of the control widget.
"""

from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal


# parameter management
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
    convergence_dot_depth: float = 0.0          # dot_depth
    convergence_dot_placement: str = 'bottom'   # dot_height


class ParameterState(QObject):
    """State manager for shared runtime variables."""
    broadcast_update = Signal(AllParameters)

    def __init__(self):
        super().__init__()
        self.config = AllParameters()

    def update_parameter(self, field_name: str, value):
        """Broadcast updates to shared parameters."""
        if hasattr(self.config, field_name):
            setattr(self.config, field_name, value)
            self.broadcast_update.emit(self.config)
