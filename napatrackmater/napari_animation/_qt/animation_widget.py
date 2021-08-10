from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton

from ..animation import Animation
from ..easing import Easing
from .frame_widget import FrameWidget
import napari


class AnimationWidget(QWidget):
    """Widget for interatviely making animations using the napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        napari viewer.

    Attributes
    ----------
    key_frames : list of dict
        List of viewer state dictionaries.
    frame : int
        Currently shown key frame.
    """

    def __init__(
        self,
        viewer: 'napari.viewer.Viewer',
        savedir: None,
        start: 0,
        end: 1000,
        parent=None,
    ):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._layout.addWidget(QLabel('Animation Wizard', parent=self))

        self.frameWidget = FrameWidget(parent=self)
        self._layout.addWidget(self.frameWidget)

        # self.captureButton = QPushButton('Capture Frame', parent=self)
        # self.captureButton.clicked.connect(self._capture_keyframe_callback)
        # self._layout.addWidget(self.captureButton)

        self._layout.addStretch(1)

        self.pathText = QLineEdit(parent=self)
        self.pathText.setText(savedir + 'Track' + '.gif')
        self._layout.addWidget(self.pathText)

        self.saveButton = QPushButton('Save Animation', parent=self)
        self.saveButton.clicked.connect(self._save_callback)
        self._layout.addWidget(self.saveButton)

        self.frameWidget.endframeSpinBox.setRange(0, end)

        self.frameWidget.endframeSpinBox.setValue(end)
        self.frameWidget.startframeSpinBox.setValue(start)
        # Create animation
        
        self.animation = Animation(viewer, savedir)
        # establish key bindings
        self._add_callbacks()

    def _add_callbacks(self):
        """Bind keys"""

        self.animation.viewer.bind_key(
            "Alt-f", self._capture_keyframe_callback, overwrite=True
        )
        self.animation.viewer.bind_key(
            "Alt-r", self._replace_keyframe_callback, overwrite=True
        )
        self.animation.viewer.bind_key(
            "Alt-d", self._delete_keyframe_callback, overwrite=True
        )

        self.animation.viewer.bind_key("Alt-a", self._key_adv_frame, overwrite=True)
        self.animation.viewer.bind_key("Alt-b", self._key_back_frame, overwrite=True)

    def _release_callbacks(self):
        """Release keys"""

        self.animation.viewer.bind_key("Alt-f", None)
        self.animation.viewer.bind_key("Alt-r", None)
        self.animation.viewer.bind_key("Alt-d", None)

        self.animation.viewer.bind_key("Alt-a", None)
        self.animation.viewer.bind_key("Alt-b", None)

    def _get_interpolation_steps(self):
        return int(self.frameWidget.stepsSpinBox.value())

    def _get_easing_function(self):
        easing_name = str(self.frameWidget.easeComboBox.currentText())
        easing_func = Easing[easing_name.upper()].value
        return easing_func

    def _set_current_frame(self):
        return self.frameWidget.startframeSpinBox.setValue(self.animation.frame)

    def _capture_keyframe_callback(self, event=None):
        """Record current key-frame"""
        start = int(self.frameWidget.startframeSpinBox.value())
        end = int(self.frameWidget.endframeSpinBox.value())
        self.animation.capture_keyframe(start, end,
            steps=self._get_interpolation_steps(), ease=self._get_easing_function()
        )
        self._set_current_frame()

    def _replace_keyframe_callback(self, event=None):
        """Replace current key-frame with new view"""
        self.animation.capture_keyframe(
            steps=self._get_interpolation_steps(),
            ease=self._get_easing_function(),
            insert=False,
        )
        self._set_current_frame()

    def _delete_keyframe_callback(self, event=None):
        """Delete current key-frame"""

        self.animation.key_frames.pop(self.animation.frame)
        self.animation.frame = (self.animation.frame - 1) % len(
            self.animation.key_frames
        )
        self.animation.set_to_keyframe(self.animation.frame)
        self._set_current_frame()

    def _key_adv_frame(self, event=None):
        """Go forwards in key-frame list"""

        new_frame = (self.animation.frame + 1) % len(self.animation.key_frames)
        self.animation.set_to_keyframe(new_frame)
        self._set_current_frame()

    def _key_back_frame(self, event=None):
        """Go backwards in key-frame list"""

        new_frame = (self.animation.frame - 1) % len(self.animation.key_frames)
        self.animation.set_to_keyframe(new_frame)
        self._set_current_frame()

    def _save_callback(self, event=None):

        """Record current key-frame"""
        self._capture_keyframe_callback()

        path = self.pathText.text()
        self.animation.animate(path)

    def close(self):
        self._release_callbacks()
        super().close()
