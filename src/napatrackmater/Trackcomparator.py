import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from typing import Union, Dict
from .Trackmate import TrackMate
import concurrent.futures
from tqdm import tqdm

class TrackComparator:
    """
    Simplified comparator: performs a linear assignment between GT and predicted tracks,
    returns the assignments and count of matches within a distance threshold.
    Uses a ThreadPoolExecutor with progress bar to parallelize cost matrix computation.
    Supports downsampling of GT tracks by a user-provided factor and computes CCA and CT.
    """
    def __init__(self,
                 gt: Union[str, 'TrackMate'],
                 pred: Union[str, 'TrackMate'],
                 downsample: int = 1):  # downsample factor for GT tracks
        # Load TrackMate instances
        self.gt = TrackMate(xml_path=gt, enhanced_computation=False) if isinstance(gt, str) else gt
        self.pred = TrackMate(xml_path=pred, enhanced_computation=False) if isinstance(pred, str) else pred
        # Cache DataFrames once
        self.gt_df = self.gt.get_track_dataframe()
        self.pred_df = self.pred.get_track_dataframe()
        # Build raw point-cloud dicts
        raw_gt = self._track_cloud(self.gt_df)
        raw_pred = self._track_cloud(self.pred_df)
        # Downsample factor (1 = no downsampling)
        self.downsample = max(1, downsample)
        # Create track items lists once
        self.gt_items = [(tid, coords[::self.downsample]) for tid, coords in raw_gt.items()]
        self.pred_items = list(raw_pred.items())

    def _track_cloud(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Group spots by unique_id and return dict of track_id -> Nx3 array."""
        return {tid: grp[['z','y','x']].values for tid, grp in df.groupby('unique_id')}

    def evaluate(self, threshold: float) -> Dict[str, object]:
        """
        Perform optimal assignment between GT and predicted tracks, using
        concurrent threads and a progress bar to build the cost matrix.

        Returns a dict with:
          - 'assignments': DataFrame with ['gt_track','pred_track','distance','matched']
          - 'num_hits': int, count of GT tracks with matched == True
          - 'num_gt': total number of GT tracks
          - 'num_pred': total number of predicted tracks
          - 'cca': float, cell cycle accuracy metric
          - 'ct': float, complete tracks metric
        """
        num_gt = len(self.gt_items)
        num_pred = len(self.pred_items)
        # Prebuild KD-trees for predicted tracks
        pred_trees = {pid: cKDTree(coords) for pid, coords in self.pred_items if coords.size}

        # Compute a single row of cost matrix
        def compute_row(item):
            i, (gt_id, gt_coords) = item
            row = np.full(num_pred, np.inf)
            if gt_coords.size:
                tree_gt = cKDTree(gt_coords)
                for j, (pred_id, pred_coords) in enumerate(self.pred_items):
                    if pred_coords.size:
                        tree_pred = pred_trees[pred_id]
                        d_gt = tree_pred.query(gt_coords)[0]
                        d_pred = tree_gt.query(pred_coords)[0]
                        row[j] = max(d_gt.mean(), d_pred.mean())
            return i, row

        # Build cost matrix in parallel with progress bar
        cost = np.full((num_gt, num_pred), np.inf)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(compute_row, item) for item in enumerate(self.gt_items)]
            for f in tqdm(concurrent.futures.as_completed(futures), total=num_gt,
                          desc="Computing distances"):
                i, row = f.result()
                cost[i, :] = row

        # Optimal assignment
        gt_idx, pred_idx = linear_sum_assignment(cost)
        records = []
        for i, j in zip(gt_idx, pred_idx):
            gt_id, _ = self.gt_items[i]
            pred_id, _ = self.pred_items[j]
            dist = float(cost[i, j])
            matched = dist <= threshold
            records.append({'gt_track': gt_id,
                            'pred_track': pred_id,
                            'distance': dist,
                            'matched': matched})
        assignments = pd.DataFrame(records)
        num_hits = int(assignments['matched'].sum())

        # Metrics
        cca = self.cca_metric()
        ct = self.ct_metric(assignments)

        return {'assignments': assignments,
                'num_hits': num_hits,
                'num_gt': num_gt,
                'num_pred': num_pred,
                'cca': cca,
                'ct': ct}

    def cca_metric(self) -> float:
        """
        Compute Cell Cycle Accuracy (CCA) between GT and predicted track-length distributions.
        Assumes no splits: uses cached DataFrames for birth/end frames.

        Returns:
            CCA score in [0,1].
        """
        # Extract birth/end for GT tracks
        gt_lengths = []
        for tid, _ in self.gt_items:
            sub = self.gt_df[self.gt_df['unique_id'] == tid]
            birth = sub['t'].min()
            end = sub['t'].max()
            gt_lengths.append(end - birth)
        # Extract birth/end for predicted tracks
        pred_lengths = []
        for tid, _ in self.pred_items:
            sub = self.pred_df[self.pred_df['unique_id'] == tid]
            birth = sub['t'].min()
            end = sub['t'].max()
            pred_lengths.append(end - birth)
        gt_lengths = np.array(gt_lengths, dtype=int)
        pred_lengths = np.array(pred_lengths, dtype=int)
        if gt_lengths.size == 0:
            return np.nan
        # Histograms
        max_len = max(gt_lengths.max(), pred_lengths.max())
        hist_gt = np.bincount(gt_lengths, minlength=max_len+1).astype(float)
        hist_gt /= hist_gt.sum()
        cum_gt = np.cumsum(hist_gt)
        hist_pred = np.bincount(pred_lengths, minlength=max_len+1).astype(float)
        hist_pred /= hist_pred.sum()
        cum_pred = np.cumsum(hist_pred)
        return float(1.0 - np.max(np.abs(cum_gt - cum_pred)))

    def ct_metric(self, assignments: pd.DataFrame) -> float:
        """
        Compute Complete Tracks (CT) metric: 2*T_rc/(T_r+T_c)
        where T_rc is number of GT tracks whose assigned pred track
        has identical birth and end times.
        """
        # Reference and predicted counts
        T_r = len(self.gt_items)
        T_c = len(self.pred_items)
        # Build dict of start/end for GT and pred
        gt_span = {tid: (
            self.gt_df[self.gt_df['unique_id']==tid]['t'].min(),
            self.gt_df[self.gt_df['unique_id']==tid]['t'].max())
            for tid, _ in self.gt_items}
        pred_span = {tid: (
            self.pred_df[self.pred_df['unique_id']==tid]['t'].min(),
            self.pred_df[self.pred_df['unique_id']==tid]['t'].max())
            for tid, _ in self.pred_items}
        # Count correctly recovered
        T_rc = 0
        for _, row in assignments.iterrows():
            gt_id = row['gt_track']
            pred_id = row['pred_track']
            if gt_span[gt_id] == pred_span[pred_id]:
                T_rc += 1
        if (T_r + T_c) == 0:
            return np.nan
        return float(2 * T_rc / (T_r + T_c))
