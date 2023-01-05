import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QScrollArea,
    QScroller
)
from qtpy import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Window
        self.setWindowTitle("DataVisualizationPrototype")
        self.setGeometry(400, 200, 900, 800)
        self.activateWindow()
        self.raise_()

        self.tab_widget = TabWidget()
        self.setCentralWidget(self.tab_widget)


class TabWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)

        self.tab1 = QWidget()
        self.plot_button = QPushButton("Add plot")
        lay = QVBoxLayout(self.tab1)
        lay.addWidget(self.plot_button)

        self.tab2 = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_container = QWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_container)
        self.scroll_layout = QHBoxLayout(self.scroll_container)
        lay = QVBoxLayout(self.tab2)
        lay.addWidget(self.scroll_area)

        self.addTab(self.tab1, "Home")
        self.addTab(self.tab2, "Comparison")

        self.plot_button.clicked.connect(self.plot)

    def plot(self):
        canvas = FigureCanvas(Figure())
        ax = canvas.figure.add_subplot(111)
        toolbar = NavigationToolbar(canvas, self)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.addWidget(canvas)
        lay.addWidget(toolbar)

        self.scroll_layout.addWidget(container)
        container.setMinimumWidth(4000)

        ax.plot([1, 2, 3, 4])
        ax.set_ylabel("some numbers")


def main():

    app = QApplication(sys.argv)
    view = MainWindow()
    view.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()