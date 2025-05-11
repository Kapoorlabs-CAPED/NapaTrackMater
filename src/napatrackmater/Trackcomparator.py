#!/usr/bin/env python3
"""
Top-level comparison of two TrackMate XMLs without CLI parsing,
computing assignment-only metrics plus CCA and CT, and summary plots.
"""
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
    Supports downsampling of GT tracks by a user-provided factor and computes CCA and CT,
    aligning predicted frame spans back to GT grid.
    """
    def __init__(self,
                 gt: Union[str, 'TrackMate'],
                 pred: Union[str, 'TrackMate'],
                 downsample: int = 1):
        # Load TrackMate instances
        self.gt = TrackMate(xml_path=gt, enhanced_computation=False) if isinstance(gt, str) else gt
        self.pred = TrackMate(xml_path=pred, enhanced_computation=False) if isinstance(pred, str) else pred
        # Cache DataFrames
        self.gt_df = self.gt.get_track_dataframe()
        self.pred_df = self.pred.get_track_dataframe()
        # Raw point-cloud dicts
        raw_gt = self._track_cloud(self.gt_df)
        raw_pred = self._track_cloud(self.pred_df)
        # Downsample factor (1=no downsampling)
        self.downsample = max(1, downsample)
        # Precompute track items
        self.gt_items = [(tid, coords[::self.downsample]) for tid, coords in raw_gt.items()]
        self.pred_items = list(raw_pred.items())

    def _track_cloud(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Group spots by unique_id and return dict of track_id -> Nx3 array."""
        return {tid: grp[['z','y','x']].values for tid, grp in df.groupby('unique_id')}

    def evaluate(self, threshold: float) -> Dict[str, object]:
        """
        Perform optimal assignment between GT and predicted tracks.
        Returns dict with:
          - assignments: DataFrame ['gt_track','pred_track','distance','matched']
          - num_hits, num_gt, num_pred, cca, ct
        """
        num_gt = len(self.gt_items)
        num_pred = len(self.pred_items)
        # KD-trees for pred tracks
        pred_trees = {pid: cKDTree(coords) for pid, coords in self.pred_items if coords.size}
        # cost matrix row function
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
        # build cost matrix
        cost = np.full((num_gt, num_pred), np.inf)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(compute_row, item) for item in enumerate(self.gt_items)]
            for f in tqdm(concurrent.futures.as_completed(futures), total=num_gt, desc="Computing distances"):
                i, row = f.result(); cost[i] = row
        # assignment
        gt_idx, pred_idx = linear_sum_assignment(cost)
        rec = []
        for i,j in zip(gt_idx,pred_idx):
            gt_id,_ = self.gt_items[i]; pred_id,_ = self.pred_items[j]
            dist = float(cost[i,j]); matched = dist<=threshold
            rec.append({'gt_track':gt_id,'pred_track':pred_id,'distance':dist,'matched':matched})
        assignments = pd.DataFrame(rec)
        num_hits = int(assignments['matched'].sum())
        # metrics
        cca = self.cca_metric()
        ct  = self.ct_metric(assignments)
        return {'assignments':assignments,'num_hits':num_hits,
                'num_gt':num_gt,'num_pred':num_pred,'cca':cca,'ct':ct}

    def cca_metric(self) -> float:
        """Cell Cycle Accuracy: CDF distance of track-length histograms."""
        # GT lengths in GT-frame units
        gt_len = [(self.gt_df[self.gt_df['unique_id']==tid]['t'].max() -
                   self.gt_df[self.gt_df['unique_id']==tid]['t'].min())
                  for tid,_ in self.gt_items]
        # Pred lengths up-sampled back to GT grid
        pred_len = [((self.pred_df[self.pred_df['unique_id']==tid]['t'].max() -
                      self.pred_df[self.pred_df['unique_id']==tid]['t'].min()) * self.downsample)
                    for tid,_ in self.pred_items]
        gt_arr = np.array(gt_len, int); pred_arr = np.array(pred_len, int)
        if gt_arr.size==0: return np.nan
        M = max(gt_arr.max(), pred_arr.max())
        h_gt = np.bincount(gt_arr, minlength=M+1).astype(float); h_gt/=h_gt.sum(); c_gt=np.cumsum(h_gt)
        h_p  = np.bincount(pred_arr, minlength=M+1).astype(float); h_p/=h_p.sum(); c_p =np.cumsum(h_p)
        return float(1 - np.max(np.abs(c_gt-c_p)))

    def ct_metric(self, assignments: pd.DataFrame) -> float:
        """Complete Tracks: fraction with matching start/end (±0 frames)."""
        T_r = len(self.gt_items); T_c = len(self.pred_items)
        # aligned spans
        spans = {}
        for tid,_ in self.gt_items:
            sub = self.gt_df[self.gt_df['unique_id']==tid]
            spans.setdefault('gt',{})[tid] = (int(sub['t'].min()), int(sub['t'].max()))
        for tid,_ in self.pred_items:
            sub = self.pred_df[self.pred_df['unique_id']==tid]
            b = int(sub['t'].min())*self.downsample; e = int(sub['t'].max())*self.downsample
            spans.setdefault('pred',{})[tid] = (b,e)
        # count T_rc
        T_rc=0
        for _,r in assignments.iterrows():
            gt_id,pr_id = r['gt_track'], r['pred_track']
            if spans['gt'][gt_id]==spans['pred'][pr_id]: T_rc+=1
        return float(2*T_rc/(T_r+T_c)) if (T_r+T_c)>0 else np.nan
