import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from typing import Union
from .Trackmate import TrackMate

# Assuming TrackMate class is available in the current module or imported appropriately
# from trackmate_module import TrackMate

class TrackComparator:
    """
    Compare ground truth and predicted TrackMate trajectories by computing
    3D track-to-track distances and optimal assignment.

    Uses symmetric average minimal distance as the pairwise metric, then
    solves a linear assignment to match predicted tracks to ground truth.
    """
    def __init__(self,
                 gt: Union[str, 'TrackMate'],
                 pred: Union[str, 'TrackMate']):
        # Load GT and prediction (paths or existing TrackMate instances)
        self.gt = TrackMate(xml_path=gt, enhanced_computation=False) if isinstance(gt, str) else gt
        self.pred = TrackMate(xml_path=pred, enhanced_computation=False) if isinstance(pred, str) else pred

        # Extract track dataframes
        self.gt_df = self.gt.get_track_dataframe()
        self.pred_df = self.pred.get_track_dataframe()

    def _track_cloud(self, df: pd.DataFrame) -> dict:
        """Group df by unique_id and return dict of track_id -> Nx3 array."""
        groups = df.groupby('unique_id')
        return {tid: grp[['z','y','x']].values for tid, grp in groups}

    def compute_pairwise_distances(self) -> pd.DataFrame:
        """
        Compute symmetric average minimal distances for all GT-pred track pairs.
        Returns a DataFrame with columns ['gt_track','pred_track','distance'].
        """
        gt_tracks = self._track_cloud(self.gt_df)
        pred_tracks = self._track_cloud(self.pred_df)
        records = []
        for gt_id, gt_coords in gt_tracks.items():
            tree_gt = cKDTree(gt_coords) if gt_coords.size else None
            for pred_id, pred_coords in pred_tracks.items():
                tree_pred = cKDTree(pred_coords) if pred_coords.size else None
                if gt_coords.size and pred_coords.size:
                    # directed distances
                    d_gt = tree_pred.query(gt_coords)[0]
                    d_pred = tree_gt.query(pred_coords)[0]
                    avg_gt = np.mean(d_gt)
                    avg_pred = np.mean(d_pred)
                    dist = max(avg_gt, avg_pred)
                else:
                    dist = np.inf
                records.append({'gt_track': gt_id,
                                'pred_track': pred_id,
                                'distance': float(dist)})
        return pd.DataFrame(records)

    def match_tracks(self) -> pd.DataFrame:
        """
        Solve optimal assignment between GT and predicted tracks minimizing total distance.
        Returns DataFrame with columns ['gt_track','pred_track','distance'] for matched pairs.
        """
        df = self.compute_pairwise_distances()
        # pivot to matrix
        pivot = df.pivot(index='gt_track', columns='pred_track', values='distance').fillna(np.inf)
        cost = pivot.values
        gt_idx, pred_idx = linear_sum_assignment(cost)
        matches = []
        gt_list = pivot.index.tolist()
        pred_list = pivot.columns.tolist()
        for i, j in zip(gt_idx, pred_idx):
            gt_id = gt_list[i]
            pred_id = pred_list[j]
            dist = cost[i, j]
            matches.append({'gt_track': gt_id,
                            'pred_track': pred_id,
                            'distance': float(dist)})
        return pd.DataFrame(matches)

    def evaluate(self) -> dict:
        """
        Full evaluation:
          - 'all_distances': DataFrame of all pairwise distances
          - 'assignment': DataFrame of optimal GT-pred assignments
        """
        all_dist = self.compute_pairwise_distances()
        assignment = self.match_tracks()
        return {
            'all_distances': all_dist,
            'assignment': assignment
        }
