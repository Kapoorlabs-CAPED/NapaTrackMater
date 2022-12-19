from qtpy.QtCore import Qt

from qtpy.QtWidgets import (
    
    QFormLayout,
    QComboBox,
    QSpinBox,
    QWidget,
    QSlider,
    QRa
)


class NapatrackmaterFrameWidget(QWidget):
    
    def __ini__(self, parent = None):
        super().__init__(parent = parent)
        
        self._layout = QFormLayout(parent = self)
        self.tracktypebox = QComboBox()
        index = self.tracktypebox.findText('linear', Qt.MatchFixedString)
        self.tracktypebox.setCurrentIndex(index)
        
        
        
        self.mindurSlider = QSlider(Qt.Horizontal, parent = self)
        self.mindurSlider.setToolTip('Ignore tracks smaller than this duration in time (sec or minutes)')
        self.mindurSlider.setRange(0, 5000)
        self.mindurSlider.setSingleStep(1)
        self.mindurSlider.setTickInterval(1)
        self.mindurSlider.setValue(0)
        
        self.mindurlabel = QLabel(self)
        self.mindurlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.mindurlabel.setMinimumWidth(80)
        self.mindurlabel.setText(f"{0:.0f}")
        
        self.minsplitSlider = QSlider(Qt.Horizontal, parent = self)
        self.minsplitSlider.setToolTip('Keep tracks that have at least these many mitosis events')
        self.minsplitSlider.setRange(0, 20)
        self.minsplitSlider.setSingleStep(1)
        self.minsplitSlider.setTickInterval(1)
        self.minsplitSlider.setValue(0)
        
        self.minsplitlabel = QLabel(self)
        self.minsplitlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.minsplitlabel.setMinimumWidth(80)
        self.minsplitlabel.setText(f"{5:.0f}")
        
        self.maxsplitSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxsplitSlider.setToolTip('Ignore tracks that have mitosis events above this number')
        self.maxsplitSlider.setRange(0, 20)
        self.maxsplitSlider.setSingleStep(1)
        self.maxsplitSlider.setTickInterval(1)
        self.maxsplitSlider.setValue(5)
        
        self.maxsplitlabel = QLabel(self)
        self.maxsplitlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxsplitlabel.setMinimumWidth(80)
        self.maxsplitlabel.setText(f"{5:.0f}")
        
        self.maxdurSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxdurSlider.setToolTip('Ignore tracks larger than this duration in time (sec or minutes)')
        self.maxdurSlider.setRange(1, 5000)
        self.maxdurSlider.setSingleStep(1)
        self.maxdurSlider.setTickInterval(1)
        self.maxdurSlider.setValue(5000)
        
        self.maxsizelabel = QLabel(self)
        self.maxsizelabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxsizelabel.setMinimumWidth(80)
        self.maxsizelabel.setText(f"{1.0:.2f}")
        
        
        self.minsizeSlider = QSlider(Qt.Horizontal, parent = self)
        self.minsizeSlider.setToolTip('Ignore tracks with mean cell size below this size (um)')
        self.minsizeSlider.setRange(0, 5000)
        self.minsizeSlider.setSingleStep(1)
        self.minsizeSlider.setTickInterval(1)
        self.minsizeSlider.setValue(0)
        
        self.minsizelabel = QLabel(self)
        self.minsizelabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.minsizelabel.setMinimumWidth(80)
        self.minsizelabel.setText(f"{0.0:.2f}")
        
        self.maxsizeSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxsizeSlider.setToolTip('Ignore tracks with mean cell size above this size (um)')
        self.maxsizeSlider.setRange(1, 10000)
        self.maxsizeSlider.setSingleStep(1)
        self.maxsizeSlider.setTickInterval(1)
        self.maxsizeSlider.setValue(1)
        
        self.maxsizelabel = QLabel(self)
        self.maxsizelabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxsizelabel.setMinimumWidth(80)
        self.maxsizelabel.setText(f"{1.0:.2f}")
        
        