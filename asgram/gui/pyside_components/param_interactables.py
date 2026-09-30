# asgram/gui/pyside_components/param_interactables.py
"""
Interactable layouts for the parameter control panel in the asgram PySide6 app.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSlider, QCheckBox, QComboBox
)
try:
    from asgram.utils.params import Params
    from asgram.gui.pyside_components.param_state import ParameterState
except ModuleNotFoundError:
    from utils.params import Params
    from gui.pyside_components.param_state import ParameterState


class SliderCust(QSlider):
    """Wrapper for QSlider to inhibit undesired scrolling behavior."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, e):
        """Patched to inhibit undesired scrolling behavior."""
        e.ignore()


class ComboBoxCust(QComboBox):
    """Wrapper for QComboBox to inhibit undesired scrolling behavior."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, e):
        """Patched to inhibit undesired scrolling behavior."""
        if self.view().isVisible():
            super().wheelEvent(e)
        else:
            e.ignore()


def dbl_slider(
        manager: ParameterState, _field: str,
        bottom: int, top: int, scale: float
):
    """Double slider for asgram parameters."""
    default = Params.model_fields[_field].default
    name = _field.replace('_', ' ').title()

    min_map = float(bottom) / scale
    max_map = float(top) / scale
    scaled_default = int(round(default * scale))

    label = QLabel(f"{name}:")

    validator = QDoubleValidator(
        bottom=min_map, top=max_map, decimals=-1,
        notation=QDoubleValidator.Notation.StandardNotation
    )

    line_edit = QLineEdit()
    line_edit.setValidator(validator)
    line_edit.setText(str(default))
    line_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    line_edit.returnPressed.connect(line_edit.clearFocus)

    min_lab, max_lab = QLabel(f"{min_map:.2f}"), QLabel(f"{max_map:.2f}")
    min_lab.setStyleSheet('font-size: 8px;')
    max_lab.setStyleSheet('font-size: 8px;')

    slider = SliderCust(Qt.Orientation.Horizontal)
    slider.setRange(bottom, top)
    slider.setValue(scaled_default)

    def _update_components():
        val = manager.config.__getattribute__(_field)
        scaled_val = int(round(val * scale))
        line_edit.blockSignals(True)
        slider.blockSignals(True)
        line_edit.setText(str(val))
        slider.setValue(scaled_val)
        line_edit.blockSignals(False)
        slider.blockSignals(False)

    def _on_line_edit_change():
        """Validated double input from line edit box."""
        text = line_edit.text()
        if text:
            manager.param_update(_field, float(text))

    def _on_slider_change(val):
        """Integer input from slider."""
        manager.param_update(_field, float(val) / scale)

    manager.broadcast_update.connect(_update_components)
    line_edit.editingFinished.connect(_on_line_edit_change)
    slider.valueChanged.connect(_on_slider_change)

    component = QVBoxLayout()
    component.addStretch(1)
    sub_comp0 = QHBoxLayout()
    sub_comp0.addWidget(label)
    sub_comp0.addWidget(line_edit)
    component.addLayout(sub_comp0)
    component.addStretch(1)
    sub_comp1 = QHBoxLayout()
    sub_comp1.addWidget(min_lab)
    sub_comp1.addWidget(slider)
    sub_comp1.addWidget(max_lab)
    component.addLayout(sub_comp1)
    component.addStretch(1)

    return component


def int_slider(
        manager: ParameterState, _field: str,
        bottom: int, top: int
):
    """Integer slider for asgram parameters."""
    default = Params.model_fields[_field].default
    name = _field.replace('_', ' ').title()

    label = QLabel(f"{name}:")

    validator = QIntValidator(bottom=bottom, top=top)

    line_edit = QLineEdit()
    line_edit.setValidator(validator)
    line_edit.setText(str(default))
    line_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    line_edit.returnPressed.connect(line_edit.clearFocus)

    min_lab, max_lab = QLabel(str(bottom)), QLabel(str(top))
    min_lab.setStyleSheet('font-size: 8px;')
    max_lab.setStyleSheet('font-size: 8px;')

    slider = SliderCust(Qt.Orientation.Horizontal)
    slider.setRange(bottom, top)
    slider.setValue(default)

    def _update_components():
        val = manager.config.__getattribute__(_field)
        line_edit.blockSignals(True)
        slider.blockSignals(True)
        line_edit.setText(str(val))
        slider.setValue(val)
        line_edit.blockSignals(False)
        slider.blockSignals(False)

    def _on_line_edit_change():
        """Validated integer input from line edit box."""
        text = line_edit.text()
        if text:
            manager.param_update(_field, int(text))

    def _on_slider_change(val):
        """Integer input from slider."""
        manager.param_update(_field, val)

    manager.broadcast_update.connect(_update_components)
    line_edit.editingFinished.connect(_on_line_edit_change)
    slider.valueChanged.connect(_on_slider_change)

    component = QVBoxLayout()
    component.addStretch(1)
    sub_comp0 = QHBoxLayout()
    sub_comp0.addWidget(label)
    sub_comp0.addWidget(line_edit)
    component.addLayout(sub_comp0)
    component.addStretch(1)
    sub_comp1 = QHBoxLayout()
    sub_comp1.addWidget(min_lab)
    sub_comp1.addWidget(slider)
    sub_comp1.addWidget(max_lab)
    component.addLayout(sub_comp1)
    component.addStretch(1)

    return component


def checkbox(manager: ParameterState, _field: str):
    """Checkbox for asgram parameters."""
    default = Params.model_fields[_field].default
    name = _field.replace('_', ' ').title()

    label = QLabel(f"{name}: {default}")

    check_box = QCheckBox()
    check_box.setChecked(default)

    def _update_components():
        state = manager.config.__getattribute__(_field)
        label.setText(f"{name}: {state}")
        check_box.blockSignals(True)
        check_box.setChecked(state)
        check_box.blockSignals(False)

    def _on_check_box_change(state):
        manager.param_update(_field, 0 != state)

    manager.broadcast_update.connect(_update_components)
    check_box.stateChanged.connect(_on_check_box_change)

    component = QVBoxLayout()
    component.addStretch(1)
    component.addWidget(label)
    component.addStretch(1)
    component.addWidget(check_box)
    component.addStretch(1)

    return component


def dropdown(manager: ParameterState, _field: str):
    """Dropdown for asgram parameters."""
    default = Params.model_fields[_field].default
    name = _field.replace('_', ' ').title()

    item_dict = Params.dropdown_lookup(_field)
    keys_list = list(item_dict.keys())
    vals_list = list(item_dict.values())
    default = keys_list[vals_list.index(default)]

    label = QLabel(f"{name}: {default}")

    combo_box = ComboBoxCust()
    combo_box.addItems(keys_list)
    combo_box.setCurrentText(default)

    def _update_components():
        text = manager.config.__getattribute__(_field)
        text = keys_list[vals_list.index(text)]

        label.setText(f"{name}: {text}")
        combo_box.blockSignals(True)
        combo_box.setCurrentText(text)
        combo_box.blockSignals(False)

    def _on_combo_box_change(text):
        manager.param_update(_field, item_dict[text])

    manager.broadcast_update.connect(_update_components)
    combo_box.currentTextChanged.connect(_on_combo_box_change)

    component = QVBoxLayout()
    component.addStretch(1)
    component.addWidget(label)
    component.addStretch(1)
    component.addWidget(combo_box)
    component.addStretch(1)

    return component
