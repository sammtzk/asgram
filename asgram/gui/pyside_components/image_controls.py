# asgram/gui/pyside_components/image_controls.py
"""
The image generation panels for the asgram PySide6 app. Establishes the image
processing pipeline.

Run this module with python -m asgram.gui.pyside_components.image_controls
"""

import sys
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QFileDialog, QPushButton
)
try:
    from asgram.gui.pyside_components.param_state import ParameterState
    from asgram.gui.pyside_components.pil_wrap import pil_to_pixmap, blank_pil
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.art import synthesizer
    from asgram.postprocessing import finish
except ModuleNotFoundError:
    from gui.pyside_components.param_state import ParameterState
    from gui.pyside_components.pil_wrap import pil_to_pixmap, blank_pil
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from art import synthesizer
    from postprocessing import finish


# image control widget
class ImageControl(QGroupBox):
    """
    Controls layout, file management, and image generation for asgram method in
    PySide6 application.
    """

    def __init__(self, manager: ParameterState):
        super().__init__("Autostereogram Creation")
        self.manager = manager

        self.depth_map_path = None
        self.zmap_instance = None

        self.source_pattern_path = None
        self.spat_instance = None

        self.asgram_mat = None
        self.asgram_pil = None

        self.final_output = None

        self._gui_init()

    # depth map processing ====================================================
    def _zmap_path_display(self):
        return f"Selected File: {self.depth_map_path}"

    def _zmap_instance_dims_display(self):
        if self.zmap_instance is not None:
            w, h = self.zmap_instance.size
            return f"Depth Map Width: {w}px ; Height: {h}px"
        elif self.depth_map_path is not None:
            return "Depth Map not yet processed"
        else:
            return "Upload a Depth Map"

    def _retrieve_zmap(self):
        zmap_path, _ = QFileDialog.getOpenFileName(
            parent=self, caption="File Select", dir="",
            filter="Supported Images (*.png *.jpg *.jpeg *.exr);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;EXR Files (*.exr)"  # noqa E501
        )

        if zmap_path:
            self.depth_map_path = zmap_path
        self.dmu_path_display.setText(self._zmap_path_display())
        self.dmm_dims.setText(self._zmap_instance_dims_display())

    def _clear_zmap_path(self):
        self.depth_map_path = None
        self.dmu_path_display.setText(self._zmap_path_display())
        self.dmm_dims.setText(self._zmap_instance_dims_display())

    def _depth_map_making(self):
        """Wraps ZMap to use shared parameter state."""
        if self.depth_map_path is not None:
            params = self.manager.config
            self.zmap_instance = ZMap(
                source=self.depth_map_path,
                mu=params.depth_of_field,
                dpi=params.dots_per_inch,
                scale=params.scale_depth_map,
                iis=params.depth_map_smoothing,
                bil=params.depth_map_bilateral_filter,
                invert=params.invert_depth_map,
                normalize=params.normalize_depth_map,
                pad=params.pad_depth_map,
                num_jobs=params.parallelization_cores
            )
            self.dmm_image.setPixmap(pil_to_pixmap(self.zmap_instance.zm_img))
        self.dmm_dims.setText(self._zmap_instance_dims_display())

    def _clear_zmap_instance(self):
        self.zmap_instance = None
        self.dmm_image.setPixmap(pil_to_pixmap(blank_pil()))
        self.dmm_dims.setText(self._zmap_instance_dims_display())

    def _view_zmap_pil(self):
        if self.zmap_instance is not None:
            self.zmap_instance.zm_img.show()

    # source pattern processing ===============================================
    def _spat_path_display(self):
        return f"Selected File: {self.source_pattern_path}"

    def _spat_instance_dims_display(self):
        if self.spat_instance is not None and self.zmap_instance is not None:
            w, h = self.spat_instance.size
            w0, h0 = self.zmap_instance.size
            if w == w0 and h == h0:
                return "Source Pattern matches Depth Map dimensions"
            else:
                return "Remake Source Pattern -- Depth Map dimensions mismatch"
        elif self.zmap_instance is not None:
            return "Source Pattern not yet processed"
        elif self.depth_map_path is not None:
            return "Depth Map not yet processed"
        else:
            return "Upload a Depth Map"

    def _retrieve_spat(self):
        spat_path, _ = QFileDialog.getOpenFileName(
            parent=self, caption="File Select", dir="",
            filter="Supported Images (*.png *.jpg *.jpeg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"  # noqa E501
        )

        if spat_path:
            self.source_pattern_path = spat_path
        self.spu_path_display.setText(self._spat_path_display())
        self.spm_dims.setText(self._spat_instance_dims_display())

    def _clear_spat_path(self):
        self.source_pattern_path = None
        self.spu_path_display.setText(self._spat_path_display())
        self.spm_dims.setText(self._spat_instance_dims_display())

    def _source_pattern_making(self):
        """Wraps SrcPat to use shared parameter state."""
        if self.zmap_instance is not None:
            ref_pat = None
            if self.source_pattern_path is not None:
                ref_pat = Image.open(self.source_pattern_path)
            params = self.manager.config
            self.spat_instance = SrcPat(
                size=self.zmap_instance.size,
                ref=ref_pat,
                cross_eyed=params.cross_view_flag,
                mu=params.depth_of_field,
                dpi=params.dots_per_inch,
                fit=params.pattern_fit,
                approach=params.constraint_approach,
                random_palette=params.random_pattern_palette
            )
            self.spm_image.setPixmap(pil_to_pixmap(self.spat_instance.sp_img))
        self.spm_dims.setText(self._spat_instance_dims_display())

    def _clear_spat_instance(self):
        self.spat_instance = None
        self.spm_image.setPixmap(pil_to_pixmap(blank_pil()))
        self.spm_dims.setText(self._spat_instance_dims_display())

    def _view_spat_pil(self):
        if self.spat_instance is not None:
            self.spat_instance.sp_img.show()

    # asgram constraints generation ===========================================
    def _constraints_generation(self):
        """Wraps synthesizer to use shared parameter state."""
        if self.zmap_instance is not None and self.spat_instance is not None:
            params = self.manager.config
            self.asgram_mat = synthesizer(
                zmap=self.zmap_instance,
                sp=self.spat_instance,
                mu=params.depth_of_field,
                dpi=params.dots_per_inch,
                cross=params.cross_view_flag,
                approach=params.constraint_approach,
                num_jobs=params.parallelization_cores
            )
            self.asgram_pil = Image.fromarray(self.asgram_mat.T)
            self.asg_image.setPixmap(pil_to_pixmap(self.asgram_pil, 'l'))

    def _clear_constraints(self):
        self.asgram_mat = None
        self.asgram_pil = None
        self.asg_image.setPixmap(pil_to_pixmap(blank_pil(), 'l'))

    def _view_asg_pil(self):
        if self.asgram_pil is not None:
            self.asgram_pil.show()

    # asgram postprocessing and finalization ==================================
    def _finalize_asgram(self):
        """Wraps finish to use shared parameter state."""
        if self.asgram_mat is not None:
            params = self.manager.config
            self.final_output = Image.fromarray(finish(
                asg=self.asgram_mat,
                depth=params.convergence_dot_depth,
                height=params.convergence_dot_placement,
                mu=params.depth_of_field,
                dpi=params.dots_per_inch,
                cross=params.cross_view_flag,
                pdvrs=params.pixel_disparity_smoothing,
                num_jobs=params.parallelization_cores
            ).T)
            self.fin_image.setPixmap(pil_to_pixmap(self.final_output, 'l'))

    def _clear_final(self):
        self.final_output = None
        self.fin_image.setPixmap(pil_to_pixmap(blank_pil(), 'l'))

    def _view_fin_pil(self):
        if self.final_output is not None:
            self.final_output.show()

    # layout ==================================================================
    def _gui_init(self):
        # start depth map processing ==========================================
        depth_map_processing_group = QGroupBox("Depth Map Processing")
        depth_map_processing_layout = QVBoxLayout()

        # depth map upload
        self.dmu = QPushButton("Upload Depth Map")
        self.dmu_clear = QPushButton("Clear Selection")
        self.dmu_path_display = QLabel(self._zmap_path_display())

        self.dmu.clicked.connect(self._retrieve_zmap)
        self.dmu_clear.clicked.connect(self._clear_zmap_path)

        self.dmu_buttons = QHBoxLayout()
        self.dmu_buttons.addWidget(self.dmu)
        self.dmu_buttons.addWidget(self.dmu_clear)
        depth_map_processing_layout.addLayout(self.dmu_buttons)
        depth_map_processing_layout.addWidget(self.dmu_path_display)

        # depth map making, displaying
        self.dmm = QPushButton("Process Depth Map")
        self.dmm_clear = QPushButton("Clear Depth Map")
        self.dmm_dims = QLabel(self._zmap_instance_dims_display())
        self.dmm_image = QLabel()
        self.dmm_image.setPixmap(pil_to_pixmap(blank_pil()))
        self.dmm_pil_viewer = QPushButton("View PIL Output")

        self.dmm.clicked.connect(self._depth_map_making)
        self.dmm_clear.clicked.connect(self._clear_zmap_instance)
        self.dmm_pil_viewer.clicked.connect(self._view_zmap_pil)

        self.dmm_buttons = QHBoxLayout()
        self.dmm_buttons.addWidget(self.dmm)
        self.dmm_buttons.addWidget(self.dmm_clear)
        depth_map_processing_layout.addLayout(self.dmm_buttons)
        depth_map_processing_layout.addWidget(self.dmm_dims)
        depth_map_processing_layout.addWidget(self.dmm_image)
        depth_map_processing_layout.addWidget(self.dmm_pil_viewer)

        # end depth map processing
        depth_map_processing_group.setLayout(depth_map_processing_layout)

        # start source pattern processing =====================================
        src_pat_processing_group = QGroupBox("Source Pattern Processing")
        src_pat_processing_layout = QVBoxLayout()

        # source pattern upload
        self.spu = QPushButton("Upload Source Pattern (Optional)")
        self.spu_clear = QPushButton("Clear Selection")
        self.spu_path_display = QLabel(self._spat_path_display())

        self.spu.clicked.connect(self._retrieve_spat)
        self.spu_clear.clicked.connect(self._clear_spat_path)

        self.spu_buttons = QHBoxLayout()
        self.spu_buttons.addWidget(self.spu)
        self.spu_buttons.addWidget(self.spu_clear)
        src_pat_processing_layout.addLayout(self.spu_buttons)
        src_pat_processing_layout.addWidget(self.spu_path_display)

        # source pattern making, displaying
        self.spm = QPushButton("Process Source Pattern")
        self.spm_clear = QPushButton("Clear Source Pattern")
        self.spm_dims = QLabel(self._spat_instance_dims_display())
        self.spm_image = QLabel()
        self.spm_image.setPixmap(pil_to_pixmap(blank_pil()))
        self.spm_pil_viewer = QPushButton("View PIL Output")

        self.spm.clicked.connect(self._source_pattern_making)
        self.spm_clear.clicked.connect(self._clear_spat_instance)
        self.spm_pil_viewer.clicked.connect(self._view_spat_pil)

        self.spm_buttons = QHBoxLayout()
        self.spm_buttons.addWidget(self.spm)
        self.spm_buttons.addWidget(self.spm_clear)
        src_pat_processing_layout.addLayout(self.spm_buttons)
        src_pat_processing_layout.addWidget(self.spm_dims)
        src_pat_processing_layout.addWidget(self.spm_image)
        src_pat_processing_layout.addWidget(self.spm_pil_viewer)

        # end source pattern processing
        src_pat_processing_group.setLayout(src_pat_processing_layout)

        # constraints generation (asgram algorithm) ===========================
        asgram_processing_group = QGroupBox("Constraints Generation")
        asgram_processing_layout = QVBoxLayout()

        self.asg = QPushButton("Generate ASGRAM Constraints")
        self.asg_clear = QPushButton("Clear ASGRAM Constraints")
        self.asg_image = QLabel()
        self.asg_image.setPixmap(pil_to_pixmap(blank_pil(), 'l'))
        self.asg_pil_viewer = QPushButton("View PIL Output")

        self.asg.clicked.connect(self._constraints_generation)
        self.asg_clear.clicked.connect(self._clear_constraints)
        self.asg_pil_viewer.clicked.connect(self._view_asg_pil)

        self.asg_buttons = QHBoxLayout()
        self.asg_buttons.addWidget(self.asg)
        self.asg_buttons.addWidget(self.asg_clear)
        asgram_processing_layout.addLayout(self.asg_buttons)
        asgram_processing_layout.addWidget(self.asg_image)
        asgram_processing_layout.addWidget(self.asg_pil_viewer)

        asgram_processing_group.setLayout(asgram_processing_layout)

        # finalize asgram (postprocessing) ====================================
        final_processing_group = QGroupBox("Postprocessing")
        final_processing_layout = QVBoxLayout()

        self.fin = QPushButton("Finalize ASGRAM")
        self.fin_clear = QPushButton("Clear Final")
        self.fin_image = QLabel()
        self.fin_image.setPixmap(pil_to_pixmap(blank_pil(), 'l'))
        self.fin_pil_viewer = QPushButton("View PIL Output")

        self.fin.clicked.connect(self._finalize_asgram)
        self.fin_clear.clicked.connect(self._clear_final)
        self.fin_pil_viewer.clicked.connect(self._view_fin_pil)

        self.fin_buttons = QHBoxLayout()
        self.fin_buttons.addWidget(self.fin)
        self.fin_buttons.addWidget(self.fin_clear)
        final_processing_layout.addLayout(self.fin_buttons)
        final_processing_layout.addWidget(self.fin_image)
        final_processing_layout.addWidget(self.fin_pil_viewer)

        final_processing_group.setLayout(final_processing_layout)

        # final formatting and placement ======================================
        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        top_layout.addWidget(depth_map_processing_group)
        top_layout.addWidget(src_pat_processing_group)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(asgram_processing_group)
        main_layout.addWidget(final_processing_group)


if __name__ == '__main__':
    # create the Qt Application
    app = QApplication(sys.argv)

    # create an application window and show it
    shared_params = ParameterState()
    window = ImageControl(shared_params)

    scroll_window = QScrollArea()
    scroll_window.setWidgetResizable(True)

    scroll_window.setWidget(window)
    scroll_window.show()

    # run the main Qt loop
    sys.exit(app.exec())
