# asgram/gui/pyside_components/param_controls.py
"""
The parameter control panel for the asgram PySide6 app. Utilizes globally
shared parameters and establishes the layout of the control widget.

Run this module with python -m asgram.gui.pyside_components.param_controls
"""

import os
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton
)
try:
    from asgram.utils.params import Params
    from asgram.gui.pyside_components.param_state import ParameterState
    from asgram.gui.pyside_components.param_interactables import (
        dbl_slider, int_slider, checkbox, dropdown
    )
    from asgram.utils.utils import _pixel_separation_summary
except ModuleNotFoundError:
    from utils.params import Params
    from gui.pyside_components.param_state import ParameterState
    from gui.pyside_components.param_interactables import (
        dbl_slider, int_slider, checkbox, dropdown
    )
    from utils.utils import _pixel_separation_summary


class ParamControl(QGroupBox):
    """
    Controls layout and state management for asgram parameters in PySide6
    application.
    """

    def __init__(self, manager: ParameterState, orientation='vertical'):
        super().__init__("Autostereogram Parameters")
        self.manager = manager
        self.orientation = orientation
        self._gui_init()

    def _load_params(self):
        params_path, _ = QFileDialog.getOpenFileName(
            parent=self, caption="File Select",
            dir="", filter="JSON Files (*.json)"
        )

        if params_path:
            load_dir, load_file = os.path.split(params_path)
            file_name, _ = os.path.splitext(load_file)
            loaded_params = Params.load(file_name, load_dir)

            for k, v in loaded_params.model_dump().items():
                self.manager.param_update(k, v)

    def _save_params(self):
        params_path, _ = QFileDialog.getSaveFileName(
            parent=self, caption="Save Autostereogram Parameters",
            dir="", filter="JSON Files (*.json)"
        )

        if params_path:
            save_dir, save_file = os.path.split(params_path)
            file_name, _ = os.path.splitext(save_file)
            self.manager.config.save(file_name, save_dir)

    def _gui_init(self):
        # start info box ======================================================
        info_box_group = QGroupBox("Pixel Constraint Summary")
        info_box_layout = QVBoxLayout()

        # labels
        self.info00 = QLabel()
        self.info01 = QLabel()
        self.info01.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.info10 = QLabel()
        self.info11 = QLabel()
        self.info11.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.info20 = QLabel()
        self.info21 = QLabel()
        self.info21.setAlignment(Qt.AlignmentFlag.AlignRight)

        def _update_info_box():
            p = self.manager.config
            info = _pixel_separation_summary(p.mu, p.dpi, p.cross)
            info = tuple(s.strip().replace('\n', '') for s in info)

            self.info00.setText(info[0])
            self.info01.setText(info[1])
            self.info10.setText(info[2])
            self.info11.setText(info[3])
            self.info20.setText(info[4])
            self.info21.setText(info[5])

        _update_info_box()
        self.manager.broadcast_update.connect(_update_info_box)

        self.info_box0 = QHBoxLayout()
        self.info_box0.addWidget(self.info00)
        self.info_box0.addWidget(self.info01)
        info_box_layout.addLayout(self.info_box0)

        self.info_box1 = QHBoxLayout()
        self.info_box1.addWidget(self.info10)
        self.info_box1.addWidget(self.info11)
        info_box_layout.addLayout(self.info_box1)

        self.info_box2 = QHBoxLayout()
        self.info_box2.addWidget(self.info20)
        self.info_box2.addWidget(self.info21)
        info_box_layout.addLayout(self.info_box2)

        # end info box
        info_box_group.setLayout(info_box_layout)

        # start fundamental ===================================================
        fundamental_group = QGroupBox("Fundamental")
        fundamental_layout = QVBoxLayout()

        # fields
        self.dof = dbl_slider(self.manager, 'depth_of_field', 1, 99, 100.0)
        fundamental_layout.addLayout(self.dof)
        self.dpi = int_slider(self.manager, 'dots_per_inch', 1, 2**12)
        fundamental_layout.addLayout(self.dpi)
        self.cvf = checkbox(self.manager, 'cross_view_flag')
        fundamental_layout.addLayout(self.cvf)
        self.ca = dropdown(self.manager, 'constraint_approach')
        fundamental_layout.addLayout(self.ca)
        self.fcg = checkbox(self.manager, 'fill_constraint_gaps')
        fundamental_layout.addLayout(self.fcg)
        self.hsr = checkbox(self.manager, 'hidden_surface_removal')
        fundamental_layout.addLayout(self.hsr)
        self.pc = int_slider(self.manager, 'parallelization_cores', -1, 20)
        fundamental_layout.addLayout(self.pc)

        # end fundamental
        fundamental_group.setLayout(fundamental_layout)

        # start depth map =====================================================
        depth_map_group = QGroupBox("Depth Map")
        depth_map_layout = QVBoxLayout()

        # fields
        self.ndm = checkbox(self.manager, 'normalize_depth_map')
        depth_map_layout.addLayout(self.ndm)
        self.idm = checkbox(self.manager, 'invert_depth_map')
        depth_map_layout.addLayout(self.idm)
        self.sdm = dbl_slider(self.manager, 'scale_depth_map', 1, 2**8, 16.0)
        depth_map_layout.addLayout(self.sdm)
        self.dmbf = checkbox(self.manager, 'depth_map_bilateral_filter')
        depth_map_layout.addLayout(self.dmbf)
        self.dms = checkbox(self.manager, 'depth_map_smoothing')
        depth_map_layout.addLayout(self.dms)
        self.pdm = checkbox(self.manager, 'pad_depth_map')
        depth_map_layout.addLayout(self.pdm)

        # end depth map
        depth_map_group.setLayout(depth_map_layout)

        # start source pattern ================================================
        source_pattern_group = QGroupBox("Source Pattern")
        source_pattern_layout = QVBoxLayout()

        # fields
        self.pf = dropdown(self.manager, 'pattern_fit')
        source_pattern_layout.addLayout(self.pf)
        self.sf = dropdown(self.manager, 'source_fit')
        source_pattern_layout.addLayout(self.sf)
        self.rpp = dropdown(self.manager, 'random_pattern_palette')
        source_pattern_layout.addLayout(self.rpp)
        self.rs = int_slider(self.manager, 'random_seed', 0, 9999)
        source_pattern_layout.addLayout(self.rs)

        # end source pattern
        source_pattern_group.setLayout(source_pattern_layout)

        # start postprocessing ================================================
        postprocessing_group = QGroupBox("Postprocessing")
        postprocessing_layout = QVBoxLayout()

        # fields
        self.pds = checkbox(self.manager, 'pixel_disparity_smoothing')
        postprocessing_layout.addLayout(self.pds)
        self.cdd = dropdown(self.manager, 'convergence_dot_depth')
        postprocessing_layout.addLayout(self.cdd)
        self.cdp = dropdown(self.manager, 'convergence_dot_placement')
        postprocessing_layout.addLayout(self.cdp)

        # end postprocessing
        postprocessing_group.setLayout(postprocessing_layout)

        # start params management =============================================
        params_management_group = QGroupBox("Manage Params")
        params_management_layout = QHBoxLayout()

        # buttons
        self.load_params = QPushButton("Load Params")
        self.load_params.clicked.connect(self._load_params)
        params_management_layout.addWidget(self.load_params)

        self.save_params = QPushButton("Save Params")
        self.save_params.clicked.connect(self._save_params)
        params_management_layout.addWidget(self.save_params)

        # end params management
        params_management_group.setLayout(params_management_layout)

        # final formatting and placement ======================================
        if 'vertical' == self.orientation:
            main_layout = QVBoxLayout(self)
            main_layout.addWidget(info_box_group)
            main_layout.addWidget(fundamental_group)
            main_layout.addWidget(depth_map_group)
            main_layout.addWidget(source_pattern_group)
            main_layout.addWidget(postprocessing_group)
            main_layout.addWidget(params_management_group)
        else:
            main_layout = QVBoxLayout(self)
            mid_layout = QHBoxLayout()

            mid_layout = QHBoxLayout()
            mid_layout.addWidget(fundamental_group)
            mid_layout.addWidget(depth_map_group)

            right_layout = QVBoxLayout()
            right_layout.addWidget(source_pattern_group)
            right_layout.addWidget(postprocessing_group)
            mid_layout.addLayout(right_layout)

            main_layout.addWidget(info_box_group)
            main_layout.addLayout(mid_layout)
            main_layout.addWidget(params_management_group)


if __name__ == '__main__':
    # create the Qt Application
    app = QApplication(sys.argv)

    # create an application window and show it
    shared_params = ParameterState()
    window = ParamControl(shared_params, 'box')
    window.show()

    # run the main Qt loop
    sys.exit(app.exec())
