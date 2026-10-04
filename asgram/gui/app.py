# asgram/gui/app.py
"""
PySide6 application for running asgram modules: tools for making polished
single image stereograms.

Run this module with python -m asgram.gui.app
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QScrollArea, QWidget, QSplitter
)
try:
    from asgram.gui.pyside_components.param_state import ParameterState
    from asgram.gui.pyside_components.param_controls import ParamControl
    from asgram.gui.pyside_components.image_controls import ImageControl
    from asgram.gui.pyside_components.eyecon import make_eyecon
except ModuleNotFoundError:
    from gui.pyside_components.param_state import ParameterState
    from gui.pyside_components.param_controls import ParamControl
    from gui.pyside_components.image_controls import ImageControl
    from gui.pyside_components.eyecon import make_eyecon


class ASGRAM(QMainWindow):
    """Application window for building asgrams."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASGRAM by SAMK")
        self.resize(1200, 900)
        self._gui_init()

    def _gui_init(self):
        self.shared_params = ParameterState()

        self.param_widget = ParamControl(self.shared_params, 'vertical')
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setMinimumWidth(300)
        self.param_scroll.setWidget(self.param_widget)

        self.image_widget = ImageControl(self.shared_params)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.param_scroll.setMinimumWidth(300)
        self.image_scroll.setWidget(self.image_widget)

        # main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.param_scroll)
        self.splitter.addWidget(self.image_scroll)
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)
        self.splitter.setSizes([300, 900])

        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.splitter)
        self.setWindowIcon(make_eyecon())


if __name__ == '__main__':
    # create the Qt Application
    app = QApplication(sys.argv)

    # create an application window and show it
    window = ASGRAM()
    window.show()

    # run the main Qt loop
    sys.exit(app.exec())
