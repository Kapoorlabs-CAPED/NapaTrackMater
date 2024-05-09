from scipy.spatial import cKDTree
import numpy as np
from .Trackvector import TrackVector


class NeighborAnalyzer(TrackVector):
    def __init__(self):
        super.__init__()

    def build_closeness_dict(self, radius):
        coords = []
        track_ids = []

        for track_id, track_data in self.unique_tracks.items():
            for entry in track_data:
                coords.append(entry[1:])  
                track_ids.append(track_id)

        coords = np.array(coords)
        track_ids = np.array(track_ids)

        tree = cKDTree(coords)
        closeness_dict = {}
        for i, (track_id, track_data) in enumerate(self.unique_tracks.items()):
            closest_indices = tree.query_ball_point(track_data[:, 1:], r=radius)  
            closest_track_ids = track_ids[np.concatenate(closest_indices)]  
            closeness_dict[track_id] = list(closest_track_ids)  

        return closeness_dict
        