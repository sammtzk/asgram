# asgram/gui/pyside_components/image_controls.py
"""
The image generation panels for the asgram PySide6 app. Establishes the image
processing pipeline.

Run this module with python -m asgram.gui.pyside_components.image_controls
"""

import os
import sys
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QLabel, QFileDialog, QPushButton
)
try:
    from asgram.gui.pyside_components.param_state import ParameterState
    import asgram.gui.pyside_components.pil_wrap as pw
    from asgram.gui.pyside_components.eyecon import make_eyecon
    from asgram.depth_map_making import ZMap
    from asgram.source_pattern_making import SrcPat
    from asgram.pixel_constraint_calculating import PixCon
    from asgram.postprocessing import Post
except ModuleNotFoundError:
    from gui.pyside_components.param_state import ParameterState
    import gui.pyside_components.pil_wrap as pw
    from gui.pyside_components.eyecon import make_eyecon
    from depth_map_making import ZMap
    from source_pattern_making import SrcPat
    from pixel_constraint_calculating import PixCon
    from postprocessing import Post


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

        self.pixcon_instance = None
        self.post_instance = None

        self._gui_init()

    # image upload box ========================================================
    @property
    def dmu_component(self):
        """QGroupBox for uploading depth map images."""
        dmu_group = QGroupBox("Depth Map")
        dmu_layout = QVBoxLayout()

        # overview
        self.dmu_overview = QLabel(
            "The depth map is the encoded 3 dimenstional object (hidden image)"
            " of an autostereogram. High resolution is preferred in all 3"
            " dimensions, so OpenEXR image formats are recommended for"
            " floating point pixel values. To make an asgram from previously"
            " calculated pixel constraints, see the Synthesizer class from the"
            " asgram.postprocessing module."
        )
        self.dmu_overview.setWordWrap(True)
        dmu_layout.addWidget(self.dmu_overview)

        # path display
        def _zmap_path_display():
            _text = self.depth_map_path
            if _text is not None:
                _text = os.path.basename(_text)
            return f"Selected File: {_text}"

        self.dmu_path_display = QLabel(_zmap_path_display())
        dmu_layout.addWidget(self.dmu_path_display)

        # buttons
        self.dmu_buttons = QHBoxLayout()

        def _retrieve_zmap():
            zmap_path, _ = QFileDialog.getOpenFileName(
                parent=self, caption="File Select", dir="",
                filter="Supported Images (*.png *.jpg *.jpeg *.exr);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;EXR Files (*.exr)"  # noqa E501
            )

            if zmap_path:
                self.depth_map_path = zmap_path
            self.dmu_path_display.setText(_zmap_path_display())

        self.dmu = QPushButton("Upload Depth Map")
        self.dmu.clicked.connect(_retrieve_zmap)
        self.dmu_buttons.addWidget(self.dmu)

        def _clear_zmap_path():
            self.depth_map_path = None
            self.dmu_path_display.setText(_zmap_path_display())

        self.dmu_clear = QPushButton("Clear Selection")
        self.dmu_clear.clicked.connect(_clear_zmap_path)
        self.dmu_buttons.addWidget(self.dmu_clear)

        dmu_layout.addLayout(self.dmu_buttons)

        dmu_group.setLayout(dmu_layout)
        return dmu_group

    @property
    def spu_component(self):
        """QGroupBox for uploading source pattern images."""
        spu_group = QGroupBox("Source Pattern")
        spu_layout = QVBoxLayout()

        # overview
        self.spu_overview = QLabel(
            "The source pattern is the image onto which pixel constraints are"
            " mapped. Some basic tiling methods are provided to manipulate the"
            " image, and it is recommended to fit the image to the pixel"
            " constraint source using the Source Fit parameter. If no pattern"
            " is uploaded, a random dot pattern will be created."
        )
        self.spu_overview.setWordWrap(True)
        spu_layout.addWidget(self.spu_overview)

        # path display
        def _spat_path_display():
            _text = self.source_pattern_path
            if _text is not None:
                _text = os.path.basename(_text)
            return f"Selected File: {_text}"

        self.spu_path_display = QLabel(_spat_path_display())
        spu_layout.addWidget(self.spu_path_display)

        # buttons
        self.spu_buttons = QHBoxLayout()

        def _retrieve_spat():
            spat_path, _ = QFileDialog.getOpenFileName(
                parent=self, caption="File Select", dir="",
                filter="Supported Images (*.png *.jpg *.jpeg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"  # noqa E501
            )

            if spat_path:
                self.source_pattern_path = spat_path
            self.spu_path_display.setText(_spat_path_display())

        self.spu = QPushButton("Upload Source Pattern")
        self.spu.clicked.connect(_retrieve_spat)
        self.spu_buttons.addWidget(self.spu)

        def _clear_spat_path():
            self.source_pattern_path = None
            self.spu_path_display.setText(_spat_path_display())

        self.spu_clear = QPushButton("Clear Selection")
        self.spu_clear.clicked.connect(_clear_spat_path)
        self.spu_buttons.addWidget(self.spu_clear)

        spu_layout.addLayout(self.spu_buttons)

        spu_group.setLayout(spu_layout)
        return spu_group

    @property
    def upload_box_component(self):
        """QGroupBox for managing uploaded images."""
        upload_box_group = QGroupBox("Image Uploads")
        upload_box_layout = QHBoxLayout()

        upload_box_sublayout = QVBoxLayout()
        upload_box_sublayout.addWidget(self.dmu_component)
        upload_box_sublayout.addWidget(self.spu_component)
        upload_box_layout.addLayout(upload_box_sublayout)

        self.eyecon_sep = QFrame()
        self.eyecon_sep.setMaximumWidth(16)
        self.eyecon_sep.setFrameShape(QFrame.Shape.VLine)
        upload_box_layout.addWidget(self.eyecon_sep)

        self.eyecon = QLabel()
        self.eyecon.setMaximumWidth(64)
        self.eyecon.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.eyecon.setPixmap(make_eyecon())
        upload_box_layout.addWidget(self.eyecon)

        upload_box_group.setLayout(upload_box_layout)
        return upload_box_group

    # depth map processing ====================================================
    def _depth_map_making(self):
        """Wraps ZMap to use shared parameter state."""
        if self.depth_map_path is not None:
            res = ZMap(
                source=self.depth_map_path,
                p=self.manager.config
            )
            self.zmap_instance = res
            self.dmm_image.setPixmap(pw.pil2pix(res.zm_img, self.dmm_image))
            self.dmm_details.setText(pw.pil_info(res.zm_img, res.zm_arr))
        else:
            info_text = "Steps To Complete:"
            step = 0
            if self.depth_map_path is None:
                step += 1
                info_text += f"\n    {step}: Upload Depth Map"
            self.dmm_image.setPixmap(
                pw.pil2pix(pw.blank_pil(info_text), self.dmm_image)
            )

    def _clear_zmap_instance(self):
        self.zmap_instance = None
        self.dmm_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.dmm_image))
        self.dmm_details.setText("")

    def _view_zmap_pil(self):
        if self.zmap_instance is not None:
            self.zmap_instance.zm_img.show()

    def _save_zmap(self):
        if self.zmap_instance is not None:
            zmap_path, _ = QFileDialog.getSaveFileName(
                parent=self, caption="Save Depth Map", dir="",
                filter="PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;EXR Files (*.exr)"  # noqa E501
            )

            if zmap_path:
                save_dir, save_file = os.path.split(zmap_path)
                file_name, extension = os.path.splitext(save_file)
                self.zmap_instance.save(file_name, save_dir, extension)

    @property
    def depth_map_component(self):
        """QGroupBox for processing depth maps."""
        depth_map_processing_group = QGroupBox("Depth Map Processing")
        depth_map_processing_layout = QVBoxLayout()
        depth_map_processing_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.dmm_buttons = QHBoxLayout()

        self.dmm = QPushButton("Process Depth Map")
        self.dmm.clicked.connect(self._depth_map_making)
        self.dmm_buttons.addWidget(self.dmm)

        self.dmm_clear = QPushButton("Clear Depth Map")
        self.dmm_clear.clicked.connect(self._clear_zmap_instance)
        self.dmm_buttons.addWidget(self.dmm_clear)

        depth_map_processing_layout.addLayout(self.dmm_buttons)

        depth_map_processing_layout.addStretch(1)
        self.dmm_image_box = QHBoxLayout()
        self.dmm_image_box.addStretch(1)

        self.dmm_image = QLabel()
        self.dmm_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dmm_image.setMaximumSize(350, 250)
        self.dmm_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.dmm_image))
        self.dmm_image_box.addWidget(self.dmm_image)

        self.dmm_image_box.addStretch(1)
        depth_map_processing_layout.addLayout(self.dmm_image_box)
        depth_map_processing_layout.addStretch(1)

        self.dmm_details = QLabel()
        self.dmm_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dmm_details.setText("")
        depth_map_processing_layout.addWidget(self.dmm_details)

        self.dms_buttons = QHBoxLayout()

        self.dms_pil_viewer = QPushButton("View PIL Output")
        self.dms_pil_viewer.clicked.connect(self._view_zmap_pil)
        self.dms_buttons.addWidget(self.dms_pil_viewer)

        self.dms_save_zmap = QPushButton("Save Depth Map")
        self.dms_save_zmap.clicked.connect(self._save_zmap)
        self.dms_buttons.addWidget(self.dms_save_zmap)

        depth_map_processing_layout.addLayout(self.dms_buttons)

        depth_map_processing_group.setLayout(depth_map_processing_layout)
        return depth_map_processing_group

    # pixel constraint calculating ============================================
    def _constraints_generation(self):
        """Wraps PixCon to use shared parameter state."""
        if self.zmap_instance is not None:
            res = PixCon(
                zmap=self.zmap_instance,
                p=self.manager.config
            )
            self.pixcon_instance = res
            self.con_image.setPixmap(pw.pil2pix(res.con_img, self.con_image))
            self.con_details.setText(pw.pil_info(res.con_img, res.con_mat))
        else:
            info_text = "Steps To Complete:"
            step = 0
            if self.depth_map_path is None:
                step += 1
                info_text += f"\n    {step}: Upload Depth Map"
            if self.zmap_instance is None:
                step += 1
                info_text += f"\n    {step}: Process Depth Map"
            self.con_image.setPixmap(
                pw.pil2pix(pw.blank_pil(info_text), self.con_image)
            )

    def _clear_constraints(self):
        self.pixcon_instance = None
        self.con_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.con_image))
        self.con_details.setText("")

    def _view_con_pil(self):
        if self.pixcon_instance is not None:
            self.pixcon_instance.con_img.show()

    def _save_constraints(self):
        if self.pixcon_instance is not None:
            con_path, _ = QFileDialog.getSaveFileName(
                parent=self, caption="Save Pixel Constraints (Matrix + JSON)",
                dir="", filter="NumPy Files (*.npy)"
            )

            if con_path:
                save_dir, save_file = os.path.split(con_path)
                file_name, _ = os.path.splitext(save_file)
                self.pixcon_instance.save(file_name, save_dir)

    @property
    def pixel_constraint_component(self):
        """QGroupBox for calculating pixel constraints."""
        pixcon_processing_group = QGroupBox("Pixel Constraint Calculating")
        pixcon_processing_layout = QVBoxLayout()
        pixcon_processing_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.con_buttons = QHBoxLayout()

        self.con = QPushButton("Calculate Pixel Constraints")
        self.con.clicked.connect(self._constraints_generation)
        self.con_buttons.addWidget(self.con)

        self.con_clear = QPushButton("Clear Pixel Constraints")
        self.con_clear.clicked.connect(self._clear_constraints)
        self.con_buttons.addWidget(self.con_clear)

        pixcon_processing_layout.addLayout(self.con_buttons)

        pixcon_processing_layout.addStretch(1)
        self.con_image_box = QHBoxLayout()
        self.con_image_box.addStretch(1)

        self.con_image = QLabel()
        self.con_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.con_image.setMaximumSize(350, 250)
        self.con_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.con_image))
        self.con_image_box.addWidget(self.con_image)

        self.con_image_box.addStretch(1)
        pixcon_processing_layout.addLayout(self.con_image_box)
        pixcon_processing_layout.addStretch(1)

        self.con_details = QLabel()
        self.con_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.con_details.setText("")
        pixcon_processing_layout.addWidget(self.con_details)

        self.pcs_buttons = QHBoxLayout()

        self.con_pil_viewer = QPushButton("View PIL Output")
        self.con_pil_viewer.clicked.connect(self._view_con_pil)
        self.pcs_buttons.addWidget(self.con_pil_viewer)

        self.con_save_pixcon = QPushButton("Save Pixel Constraints")
        self.con_save_pixcon.clicked.connect(self._save_constraints)
        self.pcs_buttons.addWidget(self.con_save_pixcon)

        pixcon_processing_layout.addLayout(self.pcs_buttons)

        pixcon_processing_group.setLayout(pixcon_processing_layout)
        return pixcon_processing_group

    # source pattern processing ===============================================
    def _source_pattern_making(self):
        """Wraps SrcPat to use shared parameter state."""
        if self.zmap_instance is not None:
            ref_pat = None
            if self.source_pattern_path is not None:
                ref_pat = Image.open(self.source_pattern_path)
            res = SrcPat(
                zm=self.zmap_instance,
                pc=self.pixcon_instance,
                p=self.manager.config,
                ref=ref_pat
            )
            self.spat_instance = res
            self.spm_image.setPixmap(pw.pil2pix(res.sp_img, self.spm_image))
            self.spm_details.setText(pw.pil_info(res.sp_img, res.sp_arr))
        else:
            info_text = "Steps To Complete:"
            step = 0
            if self.depth_map_path is None:
                step += 1
                info_text += f"\n    {step}: Upload Depth Map"
            if self.zmap_instance is None:
                step += 1
                info_text += f"\n    {step}: Process Depth Map"
            if self.source_pattern_path is None:
                step += 1
                info_text += f"\n    {step}: Upload Source Pattern (Optional)"
            self.spm_image.setPixmap(
                pw.pil2pix(pw.blank_pil(info_text), self.spm_image)
            )

    def _clear_spat_instance(self):
        self.spat_instance = None
        self.spm_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.spm_image))
        self.spm_details.setText("")

    def _view_spat_pil(self):
        if self.spat_instance is not None:
            self.spat_instance.sp_img.show()

    def _save_spat(self):
        if self.spat_instance is not None:
            spat_path, _ = QFileDialog.getSaveFileName(
                parent=self, caption="Save Source Pattern", dir="",
                filter="PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
            )

            if spat_path:
                save_dir, save_file = os.path.split(spat_path)
                file_name, extension = os.path.splitext(save_file)
                self.spat_instance.save(file_name, save_dir, extension)

    @property
    def source_pattern_component(self):
        """QGroupBox for processing source patterns."""
        src_pat_processing_group = QGroupBox("Source Pattern Processing")
        src_pat_processing_layout = QVBoxLayout()
        src_pat_processing_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.spm_buttons = QHBoxLayout()

        self.spm = QPushButton("Process Source Pattern")
        self.spm.clicked.connect(self._source_pattern_making)
        self.spm_buttons.addWidget(self.spm)

        self.spm_clear = QPushButton("Clear Source Pattern")
        self.spm_clear.clicked.connect(self._clear_spat_instance)
        self.spm_buttons.addWidget(self.spm_clear)

        src_pat_processing_layout.addLayout(self.spm_buttons)

        src_pat_processing_layout.addStretch(1)
        self.spm_image_box = QHBoxLayout()
        self.spm_image_box.addStretch(1)

        self.spm_image = QLabel()
        self.spm_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spm_image.setMaximumSize(350, 250)
        self.spm_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.spm_image))
        self.spm_image_box.addWidget(self.spm_image)

        self.spm_image_box.addStretch(1)
        src_pat_processing_layout.addLayout(self.spm_image_box)
        src_pat_processing_layout.addStretch(1)

        self.spm_details = QLabel()
        self.spm_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spm_details.setText("")
        src_pat_processing_layout.addWidget(self.spm_details)

        self.sps_buttons = QHBoxLayout()

        self.sps_pil_viewer = QPushButton("View PIL Output")
        self.sps_pil_viewer.clicked.connect(self._view_spat_pil)
        self.sps_buttons.addWidget(self.sps_pil_viewer)

        self.sps_save_spat = QPushButton("Save Source Pattern")
        self.sps_save_spat.clicked.connect(self._save_spat)
        self.sps_buttons.addWidget(self.sps_save_spat)

        src_pat_processing_layout.addLayout(self.sps_buttons)

        src_pat_processing_group.setLayout(src_pat_processing_layout)
        return src_pat_processing_group

    # finalize asgram (postprocessing) ========================================
    def _finalize_asgram(self):
        """Wraps finish to use shared parameter state."""
        if self.spat_instance is not None and self.pixcon_instance is not None:
            res = Post(
                sp=self.spat_instance,
                pc=self.pixcon_instance,
                p=self.manager.config
            )
            self.post_instance = res
            self.fin_image.setPixmap(pw.pil2pix(res.final_img, self.fin_image))
            self.fin_details.setText(pw.pil_info(res.final_img, res.final_arr))
        else:
            info_text = "Steps To Complete:"
            step = 0
            if self.depth_map_path is None:
                step += 1
                info_text += f"\n    {step}: Upload Depth Map"
            if self.zmap_instance is None:
                step += 1
                info_text += f"\n    {step}: Process Depth Map"
            if self.pixcon_instance is None:
                step += 1
                info_text += f"\n    {step}: Calculate Pixel Constraints"
            if self.source_pattern_path is None and self.spat_instance is None:
                step += 1
                info_text += f"\n    {step}: Upload Source Pattern (Optional)"
            if self.spat_instance is None:
                step += 1
                info_text += f"\n    {step}: Process Source Pattern"
            self.fin_image.setPixmap(
                pw.pil2pix(pw.blank_pil(info_text), self.fin_image)
            )

    def _clear_final(self):
        self.post_instance = None
        self.fin_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.fin_image))
        self.fin_details.setText("")

    def _view_fin_pil(self):
        if self.post_instance is not None:
            self.post_instance.final_img.show()

    def _save_final(self):
        if self.post_instance is not None:
            final_path, _ = QFileDialog.getSaveFileName(
                parent=self, caption="Save Final ASGRAM", dir="",
                filter="PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
            )

            if final_path:
                save_dir, save_file = os.path.split(final_path)
                file_name, extension = os.path.splitext(save_file)
                self.post_instance.save(file_name, save_dir, extension)

    @property
    def postprocessing_component(self):
        """QGroupBox for calculating pixel constraints."""
        final_processing_group = QGroupBox("Postprocessing")
        final_processing_layout = QVBoxLayout()
        final_processing_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.fin_buttons = QHBoxLayout()

        self.fin = QPushButton("Finalize ASGRAM")
        self.fin.clicked.connect(self._finalize_asgram)
        self.fin_buttons.addWidget(self.fin)

        self.fin_clear = QPushButton("Clear ASGRAM")
        self.fin_clear.clicked.connect(self._clear_final)
        self.fin_buttons.addWidget(self.fin_clear)

        final_processing_layout.addLayout(self.fin_buttons)

        final_processing_layout.addStretch(1)
        self.fin_image_box = QHBoxLayout()
        self.fin_image_box.addStretch(1)

        self.fin_image = QLabel()
        self.fin_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fin_image.setMaximumSize(1000, 750)
        self.fin_image.setPixmap(pw.pil2pix(pw.blank_pil(), self.fin_image))
        self.fin_image_box.addWidget(self.fin_image)

        self.fin_image_box.addStretch(1)
        final_processing_layout.addLayout(self.fin_image_box)
        final_processing_layout.addStretch(1)

        self.fin_details = QLabel()
        self.fin_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fin_details.setText("")
        final_processing_layout.addWidget(self.fin_details)

        self.fas_buttons = QHBoxLayout()

        self.fin_pil_viewer = QPushButton("View PIL Output")
        self.fin_pil_viewer.clicked.connect(self._view_fin_pil)
        self.fas_buttons.addWidget(self.fin_pil_viewer)

        self.fin_save_final = QPushButton("Save ASGRAM")
        self.fin_save_final.clicked.connect(self._save_final)
        self.fas_buttons.addWidget(self.fin_save_final)

        final_processing_layout.addLayout(self.fas_buttons)

        final_processing_group.setLayout(final_processing_layout)
        return final_processing_group

    # final formatting and placement ==========================================
    def _gui_init(self):
        main_layout = QVBoxLayout(self)

        main_layout.addWidget(self.upload_box_component)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.depth_map_component)
        top_layout.addWidget(self.pixel_constraint_component)
        top_layout.addWidget(self.source_pattern_component)
        main_layout.addLayout(top_layout)

        main_layout.addWidget(self.postprocessing_component)


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
