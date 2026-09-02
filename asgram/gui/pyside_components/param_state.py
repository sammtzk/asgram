# asgram/gui/pyside_components/param_state.py
"""
Establishes the framework for globally shared parameters.
"""

from PySide6.QtCore import QObject, Signal
try:
    from asgram.utils.utils import AllParameters
except ModuleNotFoundError:
    from utils.utils import AllParameters


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
