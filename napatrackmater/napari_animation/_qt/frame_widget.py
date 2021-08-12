from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget, QFormLayout, QSpinBox

from ..easing import Easing


class FrameWidget(QWidget):
    """Widget for interatviely making animations using the napari viewer."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QFormLayout(parent=self)
        self.stepsSpinBox = QSpinBox()
        self.stepsSpinBox.setValue(15)

        self.startframeSpinBox = QSpinBox()
        self.startframeSpinBox.setValue(0)

        self.endframeSpinBox = QSpinBox()
        self.endframeSpinBox.setValue(10)

        self.easeComboBox = QComboBox()
        self.easeComboBox.addItems([e.name.lower() for e in Easing])
        index = self.easeComboBox.findText('linear', Qt.MatchFixedString)
        self.easeComboBox.setCurrentIndex(index)

        self._layout.addRow('Steps', self.stepsSpinBox)
        self._layout.addRow('Ease', self.easeComboBox)
        self._layout.addRow('StartFrame', self.startframeSpinBox)
        self._layout.addRow('EndFrame', self.endframeSpinBox)
