from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from napatrackmater._qt.NapatrackmaterFrameWidget import NapatrackmaterFrameWidget
import napari
from qtpy.QtCore import Qt
class NapatrackmaterWidget(QWidget):
	
	def __init__(
		
		self,
		viewer: 'napari.viewer.Viewer',
		spot_csv: str,
		track_csv: str,
		edges_csv: str,
		parent = None,
	):
		
		super().__init__(parent = parent)
		self._layout = QVBoxLayout()
		self.setLayout(self._layout)
		self._layout.addWidget(
			QLabel("Napatrackmater Visualization Wizard", parent=self)
		)

		self.frameWidget = NapatrackmaterFrameWidget(parent=self)
		self._layout.addWidget(self.frameWidget)
		
		self.frameWidget.tracktypebox.addItem('Mitotic or Non Mitotic Trajectory')
		self.frameWidget.tracktypebox.addItem('Analyze both together')
		self.frameWidget.tracktypebox.addItem('Consider only Mitotic Trajectory')
		self.frameWidget.tracktypebox.addItem('Consider only Non Mitotic Trajectory')
		
if __name__=='__main__':
	
			viewer = napari.Viewer() 
			napatrackmater_widget = NapatrackmaterWidget(viewer, None, None, None)
			dock_widget = viewer.window.add_dock_widget(
			napatrackmater_widget, area="right"
		)
			viewer.window._qt_window.resizeDocks(
				[dock_widget], [1500], Qt.Horizontal
			)

			napari.run()
			