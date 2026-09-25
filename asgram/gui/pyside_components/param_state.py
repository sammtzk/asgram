# asgram/gui/pyside_components/param_state.py
"""
Establishes the framework for globally shared parameters.
"""

from PySide6.QtCore import QObject, Signal
try:
    from asgram.utils.params import Params
except ModuleNotFoundError:
    from utils.params import Params


class ParameterState(QObject):
    """State manager for shared runtime variables."""
    broadcast_update = Signal(Params)

    def __init__(self):
        super().__init__()
        self.config = Params()

    def param_update(self, field_name: str, value):
        """Broadcast updates to shared parameters."""
        if hasattr(self.config, field_name):
            setattr(self.config, field_name, value)
            self.broadcast_update.emit(self.config)
