import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from typing import Union
from .Trackmate import TrackMate
import concurrent.futures
from tqdm import tqdm

class TrackComparator:
    """
    Simplified comparator: performs a linear assignment between GT and predicted tracks,
    returns the assignments and count of matches within a distance threshold.
    Uses a ThreadPoolExecutor with progress bar to parallelize cost matrix computation.
    Supports downsampling of GT tracks by a user-provided factor.
    """
    def __init__(self,
                 gt: Union[str, 'TrackMate'],
                 pred: Union[str, 'TrackMate'],
                 downsample: int = 1):  # downsample factor for GT tracks
        # Load TrackMate instances
        self.gt = TrackMate(xml_path=gt, enhanced_computation=False) if isinstance(gt, str) else gt
        self.pred = TrackMate(xml_path=pred, enhanced_computation=False) if isinstance(pred, str) else pred
        # Prepare track point clouds
        self.gt_tracks = self._track_cloud(self.gt.get_track_dataframe())
        self.pred_tracks = self._track_cloud(self.pred.get_track_dataframe())
        # Store downsample factor (1 = no downsampling)
        self.downsample = max(1, downsample)

    def _track_cloud(self, df: pd.DataFrame) -> dict:
        """Group spots by unique_id and return dict of track_id -> Nx3 array."""
        return {tid: grp[['z','y','x']].values for tid, grp in df.groupby('unique_id')}

    def evaluate(self, threshold: float) -> dict:
        """
        Perform optimal assignment between GT and predicted tracks, using
        concurrent threads and a progress bar to build the cost matrix.

        Applies downsampling to GT tracks before computing distances.

        Returns:
          - assignments: DataFrame with ['gt_track','pred_track','distance','matched']
          - num_hits: int, count of GT tracks with matched == True
          - num_gt: total number of GT tracks (after downsampling)
        """
        # Apply downsampling to GT tracks
        gt_items = [(tid, coords[::self.downsample]) for tid, coords in self.gt_tracks.items()]
        pred_items = list(self.pred_tracks.items())
        num_gt = len(gt_items)
        num_pred = len(pred_items)

        # Prebuild KD-trees for predicted tracks
        pred_trees = {pid: cKDTree(coords) for pid, coords in pred_items if coords.size}

        # Function to compute a single GT row
        def compute_row(item):
            i, (gt_id, gt_coords) = item
            row = np.full(num_pred, np.inf)
            if gt_coords.size:
                tree_gt = cKDTree(gt_coords)
                for j, (pred_id, pred_coords) in enumerate(pred_items):
                    if pred_coords.size:
                        tree_pred = pred_trees[pred_id]
                        d_gt = tree_pred.query(gt_coords)[0]
                        d_pred = tree_gt.query(pred_coords)[0]
                        row[j] = max(d_gt.mean(), d_pred.mean())
            return i, row

        # Build cost matrix in parallel with progress bar
        cost = np.full((num_gt, num_pred), np.inf)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(compute_row, item) for item in enumerate(gt_items)]
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=num_gt,
                          desc="Computing track distances"):
                i, row = f.result()
                cost[i, :] = row

        # Solve assignment
        gt_idx, pred_idx = linear_sum_assignment(cost)
        records = []
        for i, j in zip(gt_idx, pred_idx):
            gt_id, _ = gt_items[i]
            pred_id, _ = pred_items[j]
            dist = float(cost[i, j])
            matched = dist <= threshold
            records.append({
                'gt_track': gt_id,
                'pred_track': pred_id,
                'distance': dist,
                'matched': matched
            })
        assignments = pd.DataFrame(records)
        num_hits = int(assignments['matched'].sum())

        return {
            'assignments': assignments,
            'num_hits': num_hits,
            'num_gt': num_gt
        }
