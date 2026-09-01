# asgram/gui/pyside_components/param_controls.py
"""
The parameter control panel for the asgram PySide6 app. Establishes the
framework for globally shared parameters and the layout of the control widget.

Run this module with python -m asgram.gui.pyside_components.param_controls
"""

import sys
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QComboBox
)
try:
    from asgram.utils.utils import AllParameters
except ModuleNotFoundError:
    from utils.utils import AllParameters


# parameter management
class ParameterState(QObject):
    """State manager for shared runtime variables."""
    broadcast_update = Signal(AllParameters)

    def __init__(self):
        super().__init__()
        self.config = AllParameters()

    def param_update(self, field_name: str, value):
        """Broadcast updates to shared parameters."""
        if hasattr(self.config, field_name):
            setattr(self.config, field_name, value)
            self.broadcast_update.emit(self.config)


# control templates
def labeled_dbl_slider(_s, _text, _field, min=0, max=10, scale=1.0):
    """Double slider for asgram parameters."""
    default = getattr(_s.manager.config, _field)
    label = QLabel(f"{_text}: {default:.2f}")

    tool = QSlider(Qt.Orientation.Horizontal)
    tool.setRange(min, max)
    tool.setValue(int(default * scale))
    tool.valueChanged.connect(
        lambda val: (
            _s.manager.param_update(_field, float(val) / scale),
            label.setText(f"{_text}: {float(val) / scale:.2f}")
        )
    )

    return label, tool


def labeled_int_slider(_s, _text, _field, min=0, max=10):
    """Integer slider for asgram parameters."""
    default = getattr(_s.manager.config, _field)
    label = QLabel(f"{_text}: {default}")

    tool = QSlider(Qt.Orientation.Horizontal)
    tool.setRange(min, max)
    tool.setValue(default)
    tool.valueChanged.connect(
        lambda val: (
            _s.manager.param_update(_field, val),
            label.setText(f"{_text}: {val}")
        )
    )

    return label, tool


def labeled_checkbox(_s, _text, _field):
    """Checkbox for asgram parameters."""
    default = getattr(_s.manager.config, _field)
    label = QLabel(f"{_text}: {default}")

    tool = QCheckBox()
    tool.setChecked(default)
    tool.stateChanged.connect(
        lambda state: (
            _s.manager.param_update(_field, state),
            label.setText(f"{_text}: {0 != state}")
        )
    )

    return label, tool


def labeled_dropdown(_s, _text, _field):
    """Dropdown for asgram parameters."""
    cdp_items = _s.manager.config.item_lookup(_field)
    cdp_default = next(iter(cdp_items), '')
    label = QLabel(f"{_text}: {cdp_default}")

    tool = QComboBox()
    tool.addItems(list(cdp_items.keys()))
    tool.currentTextChanged.connect(
        lambda text: (
            _s.manager.param_update(_field, cdp_items[text]),
            label.setText(f"{_text}: {text}")
        )
    )

    return label, tool


# parameter control widget
class ParamControl(QGroupBox):
    """
    Controls layout and state management for asgram parameters in PySide6
    application.
    """

    def __init__(self, manager: ParameterState):
        super().__init__("Autostereogram Parameters")
        self.manager = manager
        self._gui_init()

    def _gui_init(self):
        # start fundamental ===================================================
        fundamental_group = QGroupBox("Fundamental")
        fundamental_layout = QVBoxLayout()

        # depth_of_field
        self.dof_label, self.dof = labeled_dbl_slider(
            self, 'Depth of Field', 'depth_of_field', min=1, max=99, scale=100
        )
        fundamental_layout.addWidget(self.dof_label)
        fundamental_layout.addWidget(self.dof)

        # dots_per_inch
        self.dpi_label, self.dpi = labeled_int_slider(
            self, 'Dots per Inch', 'dots_per_inch', min=1, max=2**12
        )
        fundamental_layout.addWidget(self.dpi_label)
        fundamental_layout.addWidget(self.dpi)

        # cross_view_flag
        self.cvf_label, self.cvf = labeled_checkbox(
            self, 'Cross View', 'cross_view_flag'
        )
        fundamental_layout.addWidget(self.cvf_label)
        fundamental_layout.addWidget(self.cvf)

        # constraint_approach
        self.ca_label, self.ca = labeled_dropdown(
            self, 'Constraint Approach', 'constraint_approach'
        )
        fundamental_layout.addWidget(self.ca_label)
        fundamental_layout.addWidget(self.ca)

        # parallelization_cores
        self.pc_label, self.pc = labeled_int_slider(
            self, 'CPU Cores', 'parallelization_cores', min=-1, max=20
        )
        fundamental_layout.addWidget(self.pc_label)
        fundamental_layout.addWidget(self.pc)

        # end fundamental
        fundamental_group.setLayout(fundamental_layout)

        # start depth map =====================================================
        depth_map_group = QGroupBox("Depth Map")
        depth_map_layout = QVBoxLayout()

        # normalize_depth_map
        self.ndm_label, self.ndm = labeled_checkbox(
            self, 'Normalize', 'normalize_depth_map'
        )
        depth_map_layout.addWidget(self.ndm_label)
        depth_map_layout.addWidget(self.ndm)

        # invert_depth_map
        self.idm_label, self.idm = labeled_checkbox(
            self, 'Invert', 'invert_depth_map'
        )
        depth_map_layout.addWidget(self.idm_label)
        depth_map_layout.addWidget(self.idm)

        # depth_map_smoothing
        self.dms_label, self.dms = labeled_checkbox(
            self, 'Integrated Smoothing', 'depth_map_smoothing'
        )
        depth_map_layout.addWidget(self.dms_label)
        depth_map_layout.addWidget(self.dms)

        # depth_map_bilateral_filter
        self.dmbf_label, self.dmbf = labeled_checkbox(
            self, 'Bilateral Filter', 'depth_map_bilateral_filter'
        )
        depth_map_layout.addWidget(self.dmbf_label)
        depth_map_layout.addWidget(self.dmbf)

        # pad_depth_map
        self.pdm_label, self.pdm = labeled_checkbox(
            self, 'Add Padding', 'pad_depth_map'
        )
        depth_map_layout.addWidget(self.pdm_label)
        depth_map_layout.addWidget(self.pdm)

        # scale_depth_map
        self.sdm_label, self.sdm = labeled_dbl_slider(
            self, 'Resize', 'scale_depth_map', min=1, max=64, scale=4.0
        )
        depth_map_layout.addWidget(self.sdm_label)
        depth_map_layout.addWidget(self.sdm)

        # end depth map
        depth_map_group.setLayout(depth_map_layout)

        # start image pattern =================================================
        image_pattern_group = QGroupBox("Image Pattern")
        image_pattern_layout = QVBoxLayout()

        # pattern_fit
        self.pf_label, self.pf = labeled_dropdown(
            self, 'Pattern Fit', 'pattern_fit'
        )
        image_pattern_layout.addWidget(self.pf_label)
        image_pattern_layout.addWidget(self.pf)

        # random_pattern_palette
        self.rpp_label, self.rpp = labeled_dropdown(
            self, 'SIRDS Palette', 'random_pattern_palette'
        )
        image_pattern_layout.addWidget(self.rpp_label)
        image_pattern_layout.addWidget(self.rpp)

        # random_seed
        self.pc_label, self.pc = labeled_int_slider(
            self, 'Random Seed', 'random_seed', min=0, max=9999
        )
        image_pattern_layout.addWidget(self.pc_label)
        image_pattern_layout.addWidget(self.pc)

        # end image pattern
        image_pattern_group.setLayout(image_pattern_layout)

        # start postprocessing ================================================
        postprocessing_group = QGroupBox("Postprocessing")
        postprocessing_layout = QVBoxLayout()

        # pixel_disparity_smoothing
        self.pds_label, self.pds = labeled_checkbox(
            self, 'Pixel Disparity Smoothing', 'pixel_disparity_smoothing'
        )
        postprocessing_layout.addWidget(self.pds_label)
        postprocessing_layout.addWidget(self.pds)

        # convergence_dot_depth
        self.cdd_label, self.cdd = labeled_dropdown(
            self, 'Convergence Dot Depth', 'convergence_dot_depth'
        )
        postprocessing_layout.addWidget(self.cdd_label)
        postprocessing_layout.addWidget(self.cdd)

        # convergence_dot_placement
        self.cdp_label, self.cdp = labeled_dropdown(
            self, 'Convergence Dot Placement', 'convergence_dot_placement'
        )
        postprocessing_layout.addWidget(self.cdp_label)
        postprocessing_layout.addWidget(self.cdp)

        # end postprocessing
        postprocessing_group.setLayout(postprocessing_layout)

        # final formatting and placement ======================================
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(fundamental_group)
        main_layout.addWidget(depth_map_group)

        right_layout = QVBoxLayout()
        right_layout.addWidget(image_pattern_group)
        right_layout.addWidget(postprocessing_group)

        main_layout.addLayout(right_layout)


if __name__ == '__main__':
    # create the Qt Application
    app = QApplication(sys.argv)

    # create an application window and show it
    shared_params = ParameterState()
    window = ParamControl(shared_params)
    window.show()

    # run the main Qt loop
    sys.exit(app.exec())
