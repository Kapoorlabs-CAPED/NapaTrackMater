import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from typing import Union
from .Trackmate import TrackMate

class TrackComparator:
    """
    Simplified comparator: performs a linear assignment between GT and predicted tracks,
    returns only the assignments and count of matches within a distance threshold.
    """
    def __init__(self,
                 gt: Union[str, 'TrackMate'],
                 pred: Union[str, 'TrackMate']):
        # Load TrackMate instances
        self.gt   = TrackMate(xml_path=gt,   enhanced_computation=False) if isinstance(gt, str)   else gt
        self.pred = TrackMate(xml_path=pred, enhanced_computation=False) if isinstance(pred, str) else pred
        # Prepare track point clouds
        self.gt_tracks   = self._track_cloud(self.gt.get_track_dataframe())
        self.pred_tracks = self._track_cloud(self.pred.get_track_dataframe())

    def _track_cloud(self, df: pd.DataFrame) -> dict:
        """Group spots by unique_id and return dict of track_id -> Nx3 array."""
        return {tid: grp[['z','y','x']].values for tid, grp in df.groupby('unique_id')}

    def evaluate(self, threshold: float) -> dict:
        """
        Perform optimal assignment between GT and predicted tracks, then
        determine which GT tracks are correctly found (distance <= threshold).

        Returns:
          - assignments: DataFrame with ['gt_track','pred_track','distance','matched']
          - num_hits: int, count of GT tracks with matched == True
          - num_gt: total number of GT tracks
        """
        # Build cost matrix
        gt_ids   = list(self.gt_tracks.keys())
        pred_ids = list(self.pred_tracks.keys())
        cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
        for i, gt_id in enumerate(gt_ids):
            gt_coords = self.gt_tracks[gt_id]
            tree_gt = cKDTree(gt_coords)
            for j, pred_id in enumerate(pred_ids):
                pred_coords = self.pred_tracks[pred_id]
                tree_pred = cKDTree(pred_coords)
                # symmetric avg minimal distance
                d_gt   = tree_pred.query(gt_coords)[0]
                d_pred = tree_gt.query(pred_coords)[0]
                cost[i, j] = max(d_gt.mean(), d_pred.mean())

        # Optimal assignment
        gt_idx, pred_idx = linear_sum_assignment(cost)
        records = []
        for i, j in zip(gt_idx, pred_idx):
            gt_id = gt_ids[i]
            pred_id = pred_ids[j]
            dist = float(cost[i, j])
            matched = dist <= threshold
            records.append({'gt_track': gt_id,
                            'pred_track': pred_id,
                            'distance': dist,
                            'matched': matched})
        assignments = pd.DataFrame(records)

        num_hits = int(assignments['matched'].sum())
        num_gt = len(gt_ids)
        return {
            'assignments': assignments,
            'num_hits': num_hits,
            'num_gt': num_gt
        }
