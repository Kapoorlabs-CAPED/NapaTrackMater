from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

from tqdm import tqdm

import imageio
import numpy as np
import skimage.io
import skimage.transform

from .utils import interpolate_state


class Animation:
    """Make animations using the napari viewer.

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

    def __init__(self, viewer, savedir):
        self.viewer = viewer
        self.savedir = savedir
        
        self.key_frames = []
        self.frame = -1

    def capture_keyframe(self, start, end, steps=1, ease=None, insert=True, frame=None):
        """Record current key-frame

        Parameters
        ----------
        steps : int
            Number of interpolation steps between last keyframe and captured one.
        ease : callable, optional
            If provided this method should make from `[0, 1]` to `[0, 1]` and will
            be used as an easing function for the transition between the last state
            and captured one.
        insert : bool
            If captured key-frame should insert into current list or replace the current
            keyframe.
        frame : int, optional
            If provided use this value for frame rather than current frame number.
        """
        self.key_frames = []
        for frame in tqdm(range(int(start), int(end))):
            self.frame = frame
           
            new_state = {
                'viewer': self._get_viewer_state(),
                'steps': steps,
                'ease': ease,
            }
            self.key_frames.insert(self.frame + 1, new_state)
            self.viewer.dims.set_point(0, self.frame)

    def set_to_keyframe(self, frame):
        """Set the viewer to a given key-frame

        Parameters
        -------
        frame : int
            Key-frame to visualize
        """
        self.frame = frame
        if len(self.key_frames) > 0 and self.frame > -1:
            self._set_viewer_state(self.key_frames[frame])

    def _get_viewer_state(self):
        """Capture current viewer state

        Returns
        -------
        new_state : dict
            Description of viewer state.
        """
        new_state = {
            'camera': self.viewer.camera.dict(),
            'dims': self.viewer.dims.dict(),
        }
        # Log transform zoom for linear interpolation
        new_state['camera']['zoom'] = np.log10(new_state['camera']['zoom'])
        return new_state

    def _set_viewer_state(self, state):
        """Sets the current viewer state

        Parameters
        ----------
        state : dict
            Description of viewer state.
        """
        # Undo log transform zoom for linear interpolation
        camera_state = deepcopy(state['camera'])
        camera_state['zoom'] = np.power(10, camera_state['zoom'])

        self.viewer.camera.update(camera_state)
        self.viewer.dims.update(state['dims'])

    def _state_generator(self):
        if len(self.key_frames) < 2:
            raise ValueError(
                f'Must have at least 2 key frames, received {len(self.key_frames)}'
            )
        for frame in range(len(self.key_frames) - 1):
            initial_state = self.key_frames[frame]["viewer"]
            final_state = self.key_frames[frame + 1]["viewer"]
            interpolation_steps = self.key_frames[frame + 1]["steps"]
            ease = self.key_frames[frame + 1]["ease"]
            for interp in range(interpolation_steps):
                fraction = interp / interpolation_steps
                if ease is not None:
                    fraction = ease(fraction)
                state = interpolate_state(initial_state, final_state, fraction)
                yield state

    def _frame_generator(self, canvas_only=True):
        total = np.sum([f["steps"] for f in self.key_frames[1:]])
        for i, state in tqdm(enumerate(self._state_generator())):
            self._set_viewer_state(state)
            frame = self.viewer.screenshot(canvas_only=canvas_only)
            yield frame

    def animate(
        self,
        path,
        fps=20,
        quality=5,
        format=None,
        canvas_only=True,
        scale_factor=None,
    ):
        """Create a movie based on key-frames

        Parameters
        -------
        path : str
            path to use for saving the movie (can also be a path)
            should be either .mp4 or .gif. If no extension is provided,
            images are saved as a folder of PNGs
        interpolation_steps : int
            Number of steps for interpolation.
        fps : int
            frames per second
        quality: float
            number from 1 (lowest quality) to 9
            only applies to mp4
        format: str
            The format to use to write the file. By default imageio selects the appropriate for you based on the filename.
        canvas_only : bool
            If True include just includes the canvas, otherwise include the full napari viewer.
        scale_factor : float
            Rescaling factor for the image size. Only used without
            viewer (with_viewer = False).
        """

        # create a frame generator
        frame_gen = self._frame_generator(canvas_only=canvas_only)

        # create path object
        path_obj = Path(path)

        # try to create an ffmpeg writer. If not installed default to folder creation
        writer = imageio.get_writer(path, fps=fps, format=format)

        # save frames
        for ind, frame in enumerate(frame_gen):
            if scale_factor is not None:
                frame = skimage.transform.rescale(
                    frame, scale_factor, multichannel=True, preserve_range=True
                )
                frame = frame.astype(np.uint8)
            writer.append_data(frame)

        writer.close()
