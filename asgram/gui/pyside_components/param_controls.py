# asgram/gui/pyside_components/param_controls.py
"""
The parameter control panel for the asgram PySide6 app. Utilizes globally
shared parameters and establishes the layout of the control widget.

Run this module with python -m asgram.gui.pyside_components.param_controls
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QComboBox
)
try:
    from asgram.gui.pyside_components.param_state import ParameterState
except ModuleNotFoundError:
    from gui.pyside_components.param_state import ParameterState


class ParamControl(QGroupBox):
    """
    Controls layout and state management for asgram parameters in PySide6
    application.
    """

    def __init__(self, manager: ParameterState, orientation='box'):
        super().__init__("Autostereogram Parameters")
        self.manager = manager
        self.orientation = orientation
        self._gui_init()

    def labeled_dbl_slider(self, _text, _field, min=0, max=10, scale=1.0):
        """Double slider for asgram parameters."""
        default = getattr(self.manager.config, _field)
        label = QLabel(f"{_text}: {default:.2f}")

        tool = QSlider(Qt.Orientation.Horizontal)
        tool.setRange(min, max)
        tool.setValue(int(default * scale))
        tool.valueChanged.connect(
            lambda val: (
                self.manager.param_update(_field, float(val) / scale),
                label.setText(f"{_text}: {float(val) / scale:.2f}")
            )
        )

        return label, tool

    def labeled_int_slider(self, _text, _field, min=0, max=10):
        """Integer slider for asgram parameters."""
        default = getattr(self.manager.config, _field)
        label = QLabel(f"{_text}: {default}")

        tool = QSlider(Qt.Orientation.Horizontal)
        tool.setRange(min, max)
        tool.setValue(default)
        tool.valueChanged.connect(
            lambda val: (
                self.manager.param_update(_field, val),
                label.setText(f"{_text}: {val}")
            )
        )

        return label, tool

    def labeled_checkbox(self, _text, _field):
        """Checkbox for asgram parameters."""
        default = getattr(self.manager.config, _field)
        label = QLabel(f"{_text}: {default}")

        tool = QCheckBox()
        tool.setChecked(default)
        tool.stateChanged.connect(
            lambda state: (
                self.manager.param_update(_field, state),
                label.setText(f"{_text}: {0 != state}")
            )
        )

        return label, tool

    def labeled_dropdown(self, _text, _field):
        """Dropdown for asgram parameters."""
        cdp_items = self.manager.config.item_lookup(_field)
        cdp_default = next(iter(cdp_items), '')
        label = QLabel(f"{_text}: {cdp_default}")

        tool = QComboBox()
        tool.addItems(list(cdp_items.keys()))
        tool.currentTextChanged.connect(
            lambda text: (
                self.manager.param_update(_field, cdp_items[text]),
                label.setText(f"{_text}: {text}")
            )
        )

        return label, tool

    def _gui_init(self):
        # start fundamental ===================================================
        fundamental_group = QGroupBox("Fundamental")
        fundamental_layout = QVBoxLayout()

        # depth_of_field
        self.dof_label, self.dof = self.labeled_dbl_slider(
            'Depth of Field', 'depth_of_field', min=1, max=99, scale=100
        )
        fundamental_layout.addWidget(self.dof_label)
        fundamental_layout.addWidget(self.dof)

        # dots_per_inch
        self.dpi_label, self.dpi = self.labeled_int_slider(
            'Dots per Inch', 'dots_per_inch', min=1, max=2**12
        )
        fundamental_layout.addWidget(self.dpi_label)
        fundamental_layout.addWidget(self.dpi)

        # cross_view_flag
        self.cvf_label, self.cvf = self.labeled_checkbox(
            'Cross View', 'cross_view_flag'
        )
        fundamental_layout.addWidget(self.cvf_label)
        fundamental_layout.addWidget(self.cvf)

        # constraint_approach
        self.ca_label, self.ca = self.labeled_dropdown(
            'Constraint Approach', 'constraint_approach'
        )
        fundamental_layout.addWidget(self.ca_label)
        fundamental_layout.addWidget(self.ca)

        # parallelization_cores
        self.pc_label, self.pc = self.labeled_int_slider(
            'CPU Cores', 'parallelization_cores', min=-1, max=20
        )
        fundamental_layout.addWidget(self.pc_label)
        fundamental_layout.addWidget(self.pc)

        # end fundamental
        fundamental_group.setLayout(fundamental_layout)

        # start depth map =====================================================
        depth_map_group = QGroupBox("Depth Map")
        depth_map_layout = QVBoxLayout()

        # normalize_depth_map
        self.ndm_label, self.ndm = self.labeled_checkbox(
            'Normalize', 'normalize_depth_map'
        )
        depth_map_layout.addWidget(self.ndm_label)
        depth_map_layout.addWidget(self.ndm)

        # invert_depth_map
        self.idm_label, self.idm = self.labeled_checkbox(
            'Invert', 'invert_depth_map'
        )
        depth_map_layout.addWidget(self.idm_label)
        depth_map_layout.addWidget(self.idm)

        # depth_map_smoothing
        self.dms_label, self.dms = self.labeled_checkbox(
            'Integrated Smoothing', 'depth_map_smoothing'
        )
        depth_map_layout.addWidget(self.dms_label)
        depth_map_layout.addWidget(self.dms)

        # depth_map_bilateral_filter
        self.dmbf_label, self.dmbf = self.labeled_checkbox(
            'Bilateral Filter', 'depth_map_bilateral_filter'
        )
        depth_map_layout.addWidget(self.dmbf_label)
        depth_map_layout.addWidget(self.dmbf)

        # pad_depth_map
        self.pdm_label, self.pdm = self.labeled_checkbox(
            'Add Padding', 'pad_depth_map'
        )
        depth_map_layout.addWidget(self.pdm_label)
        depth_map_layout.addWidget(self.pdm)

        # scale_depth_map
        self.sdm_label, self.sdm = self.labeled_dbl_slider(
            'Resize', 'scale_depth_map', min=1, max=64, scale=4.0
        )
        depth_map_layout.addWidget(self.sdm_label)
        depth_map_layout.addWidget(self.sdm)

        # end depth map
        depth_map_group.setLayout(depth_map_layout)

        # start image pattern =================================================
        image_pattern_group = QGroupBox("Image Pattern")
        image_pattern_layout = QVBoxLayout()

        # pattern_fit
        self.pf_label, self.pf = self.labeled_dropdown(
            'Pattern Fit', 'pattern_fit'
        )
        image_pattern_layout.addWidget(self.pf_label)
        image_pattern_layout.addWidget(self.pf)

        # random_pattern_palette
        self.rpp_label, self.rpp = self.labeled_dropdown(
            'SIRDS Palette', 'random_pattern_palette'
        )
        image_pattern_layout.addWidget(self.rpp_label)
        image_pattern_layout.addWidget(self.rpp)

        # random_seed
        self.pc_label, self.pc = self.labeled_int_slider(
            'Random Seed', 'random_seed', min=0, max=9999
        )
        image_pattern_layout.addWidget(self.pc_label)
        image_pattern_layout.addWidget(self.pc)

        # end image pattern
        image_pattern_group.setLayout(image_pattern_layout)

        # start postprocessing ================================================
        postprocessing_group = QGroupBox("Postprocessing")
        postprocessing_layout = QVBoxLayout()

        # pixel_disparity_smoothing
        self.pds_label, self.pds = self.labeled_checkbox(
            'Pixel Disparity Smoothing', 'pixel_disparity_smoothing'
        )
        postprocessing_layout.addWidget(self.pds_label)
        postprocessing_layout.addWidget(self.pds)

        # convergence_dot_depth
        self.cdd_label, self.cdd = self.labeled_dropdown(
            'Convergence Dot Depth', 'convergence_dot_depth'
        )
        postprocessing_layout.addWidget(self.cdd_label)
        postprocessing_layout.addWidget(self.cdd)

        # convergence_dot_placement
        self.cdp_label, self.cdp = self.labeled_dropdown(
            'Convergence Dot Placement', 'convergence_dot_placement'
        )
        postprocessing_layout.addWidget(self.cdp_label)
        postprocessing_layout.addWidget(self.cdp)

        # end postprocessing
        postprocessing_group.setLayout(postprocessing_layout)

        # final formatting and placement ======================================
        if 'vertical' == self.orientation:
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(fundamental_group)
            main_layout.addWidget(depth_map_group)
            main_layout.addWidget(image_pattern_group)
            main_layout.addWidget(postprocessing_group)
        else:
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
    window = ParamControl(shared_params, 'box')
    window.show()

    # run the main Qt loop
    sys.exit(app.exec())
