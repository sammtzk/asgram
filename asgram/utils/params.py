# asgram/utils/params.py
"""
Dataclass for standardization and type validation of asgram input parameters.
"""

import os
import re
from typing import Literal
from pydantic import (
    BaseModel, field_validator, PositiveFloat, PositiveInt, ValidationError
)
from matplotlib import colormaps


class Params(BaseModel):
    """Parameters for autostereogram creation."""
    # fundamental
    depth_of_field: PositiveFloat = 1/3
    dots_per_inch: PositiveInt = 72
    cross_view_flag: bool = False
    constraint_approach: Literal['rl', 'lr', 'mo', 'oi', 'random'] = 'rl'
    fill_constraint_gaps: bool = False
    hidden_surface_removal: bool = True
    parallelization_cores: int = 8

    # depth map
    normalize_depth_map: bool = True
    invert_depth_map: bool = False
    scale_depth_map: PositiveFloat = 1.0
    depth_map_bilateral_filter: bool = False
    depth_map_smoothing: bool = False
    pad_depth_map: bool = False

    # image pattern
    pattern_fit: Literal['fit', 'tile', 'htile', 'vtile'] | str = 'fit'
    source_fit: Literal['estimate', 'exact', ''] | str = 'estimate'
    random_pattern_palette: Literal['bw'] | str = 'bw'
    random_seed: int = 1132

    # postprocessing
    pixel_disparity_smoothing: bool = False
    convergence_dot_depth: float = -1.0
    convergence_dot_placement: Literal['top', 'center', 'bottom'] = 'top'

    # validation config
    model_config = {'validate_assignment': True}

    @property
    def mu(self):
        """Alternate way to access self.depth_of_field."""
        return self.depth_of_field

    @property
    def dpi(self):
        """Alternate way to access self.dots_per_inch."""
        return self.dots_per_inch

    @property
    def cross(self):
        """Alternate way to access self.cross_view_flag."""
        return self.cross_view_flag

    @property
    def approach(self):
        """Alternate way to access self.constraint_approach."""
        return self.constraint_approach

    @property
    def fill(self):
        """Alternate way to access self.fill_constraint_gaps."""
        return self.fill_constraint_gaps

    @property
    def hsr(self):
        """Alternate way to access self.hidden_surface_removal."""
        return self.hidden_surface_removal

    @property
    def num_jobs(self):
        """Alternate way to access self.parallelization_cores."""
        return self.parallelization_cores

    @field_validator('pattern_fit')
    @classmethod
    def validate_pattern_fit(cls, v: str) -> str:
        """Validator for assignment to self.pattern_fit."""
        if v in ('fit', 'tile', 'htile', 'vtile'):
            return v
        if re.fullmatch(r'tile=(\d+)x(\d+)', v):
            return v
        return 'fit'

    @field_validator('source_fit')
    @classmethod
    def validate_source_fit(cls, v: str) -> str:
        """Validator for assignment to self.source_fit."""
        if v in ('estimate', 'exact'):
            return v
        return ''

    @field_validator('random_pattern_palette')
    @classmethod
    def validate_random_pattern_palette(cls, v: str) -> str:
        """Validator for assignment to self.random_pattern_palette."""
        if ('bw' != v) and (v not in set(colormaps)):
            return 'bw'
        return v

    def dropdown_lookup(self, field_name):
        """
        Dropdown menus are used for some parameter inputs in asgram guis, even
        for some data types which are not Literals. In these cases, this method
        provides keys in the form of readable display text and values in the
        form of predetermined acceptable parameter inputs for a select number
        of parameters. If a parameter is not a field of Params or has no
        associated lookup dictionary, then this method returns an empty dict.
        """
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
                        'Vertical Tile': 'vtile'
                    }
                case 'source_fit':
                    return {
                        'Estimate Source': 'estimate',
                        'Exact Source': 'exact',
                        'None': ''
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

    @classmethod
    def _parse_path(cls, file_name='temp', dir_path=''):
        dir_path = dir_path if dir_path else os.getcwd()
        file_name += '.json' if '.json' != file_name[-5:] else ''
        return os.path.join(dir_path, file_name)

    def save(self, file_name='temp', dir_path=''):
        """Saves the Params instance to .json."""
        file_path = self._parse_path(file_name, dir_path)
        try:
            with open(file_path, 'w') as f:
                f.write(self.model_dump_json(indent=4))
            print(f"Params saved to {file_path}")
        except OSError as e:
            raise OSError(f"Failed to write file to {file_path}: {e}") from e

    @classmethod
    def load(cls, file_name='temp', dir_path=''):
        """Loads an Params instance from .json."""
        file_path = cls._parse_path(file_name, dir_path)
        try:
            with open(file_path, 'r') as f:
                raw = f.read()
        except OSError as e:
            raise OSError(f"Failed to read file at {file_path}: {e}") from e
        try:
            validated = cls.model_validate_json(raw)
            print(f"Params loaded from {file_path}")
            return validated
        except ValidationError as e:
            raise ValueError(f"Data is corrupted at {file_path}: {e}") from e
