from qtpy.QtCore import Qt

from qtpy.QtWidgets import (
    
    QFormLayout,
    QComboBox,
    QSpinBox,
    QWidget,
    QSlider,
    QGridLayout,
    QTableWidget,
    QGroupBox,
    QLabel
)


class NapatrackmaterFrameWidget(QWidget):
    
    def __init__(self, parent = None):
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
        
        self.maxdurSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxdurSlider.setToolTip('Ignore tracks larger than this duration in time (sec or minutes)')
        self.maxdurSlider.setRange(0, 5000)
        self.maxdurSlider.setSingleStep(1)
        self.maxdurSlider.setTickInterval(1)
        self.maxdurSlider.setValue(0)
        
        self.maxdurlabel = QLabel(self)
        self.maxdurlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxdurlabel.setMinimumWidth(80)
        self.maxdurlabel.setText(f"{0:.0f}")
        
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
        
        self.minsizelabel = QLabel( self)
        self.minsizelabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.minsizelabel.setMinimumWidth(80)
        self.minsizelabel.setText(f"{0.0:.2f}")
        
        self.maxsizeSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxsizeSlider.setToolTip('Ignore tracks with mean cell size above this size (um)')
        self.maxsizeSlider.setRange(1, 10000)
        self.maxsizeSlider.setSingleStep(1)
        self.maxsizeSlider.setTickInterval(1)
        self.maxsizeSlider.setValue(1)
        
        self.maxsizelabel = QLabel( self)
        self.maxsizelabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxsizelabel.setMinimumWidth(80)
        self.maxsizelabel.setText(f"{1.0:.2f}")
        
        self.minspeedSlider = QSlider(Qt.Horizontal, parent = self)
        self.minspeedSlider.setToolTip('Ignore tracks with mean speed below this size (um/t)')
        self.minspeedSlider.setRange(0, 2)
        self.minspeedSlider.setSingleStep(1)
        self.minspeedSlider.setTickInterval(1)
        self.minspeedSlider.setValue(0)
        
        self.minspeedlabel = QLabel( self)
        self.minspeedlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.minspeedlabel.setMinimumWidth(80)
        self.minspeedlabel.setText(f"{0.0:.5f}")
        
        self.maxspeedSlider = QSlider(Qt.Horizontal, parent = self)
        self.maxspeedSlider.setToolTip('Ignore tracks with mean speed above this value (um/t)')
        self.maxspeedSlider.setRange(0, 2)
        self.maxspeedSlider.setSingleStep(1)
        self.maxspeedSlider.setTickInterval(1)
        self.maxspeedSlider.setValue(1)
        
        self.maxspeedlabel = QLabel( self)
        self.maxspeedlabel.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.maxspeedlabel.setMinimumWidth(80)
        self.maxspeedlabel.setText(f"{1.0:.5f}")
        
        
        self._layout.addRow(self.tracktypebox)
        
        
        self._layout.addRow('Minumum track duration (t)', self.mindurSlider)
        self._layout.addRow(self.mindurlabel)
        
        self._layout.addRow('Maximum track duration (t)',self.maxdurSlider)
        self._layout.addRow(self.maxdurlabel)
        
        self._layout.addRow('Minumum number of mitosis events', self.minsplitSlider)
        self._layout.addRow(self.minsplitlabel)
        
        self._layout.addRow('Maximum number of mitosis events', self.maxsplitSlider)
        self._layout.addRow(self.maxsplitlabel)
        
        self._layout.addRow('Minimum cell size (um)', self.minsizeSlider)
        self._layout.addRow(self.minsizelabel)
        
        self._layout.addRow('Maximum cell size (um)',self.maxsizeSlider)
        self._layout.addRow(self.maxsizelabel)
        
        self._layout.addRow('Minumum median cell speed (um/t)',self.minspeedSlider)
        self._layout.addRow(self.minspeedlabel)
        
       
        self._layout.addRow('Maximum median cell speed (um/t)', self.maxspeedSlider)
        self._layout.addRow(self.maxspeedlabel)
        
        
        
        