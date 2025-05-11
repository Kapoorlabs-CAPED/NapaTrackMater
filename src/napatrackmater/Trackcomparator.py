#!/usr/bin/env python3
"""
Top-level comparison of two TrackMate XMLs without CLI parsing,
computing assignment-only metrics plus CCA and CT, and summary plots.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple
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
                 downsampleT: int = 1):
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
        self.downsampleT = max(1, downsampleT)
        # Precompute track items
        self.gt_items = [(tid, coords) for tid, coords in raw_gt.items()]
        self.pred_items = list(raw_pred.items())

    def _track_cloud(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Group spots by unique_id and return dict of track_id -> Nx3 array."""
        return {tid: grp[['z','y','x']].values for tid, grp in df.groupby('unique_id')}

    def evaluate(self, threshold: float, compute_bci: bool = False) -> Dict[str, object]:
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
        bci = self.bci_metric(assignments) if compute_bci else None

        return {'assignments':assignments,'num_hits':num_hits,
                'num_gt':num_gt,'num_pred':num_pred,'cca':cca,'ct':ct,
            'bci': bci}

    def cca_metric(self) -> float:
        """Cell Cycle Accuracy: CDF distance of track-length histograms."""
        # GT lengths in GT-frame units
        gt_len = [(self.gt_df[self.gt_df['unique_id']==tid]['t'].max() -
                   self.gt_df[self.gt_df['unique_id']==tid]['t'].min())
                  for tid,_ in self.gt_items]
        # Pred lengths up-sampled back to GT grid
        pred_len = [((self.pred_df[self.pred_df['unique_id']==tid]['t'].max() -
                      self.pred_df[self.pred_df['unique_id']==tid]['t'].min()) * self.downsampleT)
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
            b = int(sub['t'].min())*self.downsampleT; e = int(sub['t'].max())*self.downsampleT
            spans.setdefault('pred',{})[tid] = (b,e)
        # count T_rc
        T_rc=0
        for _,r in assignments.iterrows():
            gt_id,pr_id = r['gt_track'], r['pred_track']
            if spans['gt'][gt_id]==spans['pred'][pr_id]: T_rc+=1
        return float(2*T_rc/(T_r+T_c)) if (T_r+T_c)>0 else np.nan
    
    def _get_mitosis_events(self, tm: TrackMate) -> List[Tuple[int,int]]:
        """
        Walk every dividing track in tm and return a list of
        (track_id, split_frame) for *every* split event.
        """
        events = []
        for trk in tm.DividingTrackIds:
            if trk is None or trk == tm.TrackidBox:
                continue
            for spot in tm.all_current_cell_ids[int(trk)]:
                children = tm.edge_target_lookup.get(spot, [])
                # split if >1 children
                if isinstance(children, list) and len(children) > 1:
                    t_split = tm.unique_spot_properties[spot][tm.frameid_key]
                    events.append((int(trk), int(t_split)))
                    # no break → capture multiple splits per track
        return events

    def bci_metric(self,
                   assignments: pd.DataFrame,
                   tol: int = 1
    ) -> float:
        """
        Branching Correctness Index (F1) of mitotic events.
        Matches every GT (track,time) split to its assigned pred track within ±tol frames.
        """
        gt_events   = self._get_mitosis_events(self.gt)
        pred_events = self._get_mitosis_events(self.pred)

        # map GT → assigned pred
        assign_map = {
            int(r.gt_track): int(r.pred_track)
            for r in assignments.itertuples(index=False)
            if r.matched
        }

        tp = fp = fn = 0

        # true / false negatives
        for gt_tid, t_gt in gt_events:
            pr_tid = assign_map.get(gt_tid)
            if pr_tid is None:
                fn += 1
            else:
                # did that pred track also split near the same time?
                if any(pr == pr_tid and abs(t_pr - t_gt) <= tol
                       for pr, t_pr in pred_events):
                    tp += 1
                else:
                    fn += 1

        # false positives: pred splits that didn’t match any GT
        matched_preds = set(assign_map.values())
        for pr_tid, t_pr in pred_events:
            # if this pred track wasn’t assigned at all → FP
            if pr_tid not in matched_preds:
                fp += 1
            else:
                # if its matched GT track never split at this time → FP
                # find all GT tracks that map to this pred
                gt_tids = [gt for gt, pr in assign_map.items() if pr == pr_tid]
                # for any such GT, is there a GT split within tol of t_pr?
                matched_any = False
                for candidate_gt in gt_tids:
                    for gt2, t_gt2 in gt_events:
                        if gt2 == candidate_gt and abs(t_pr - t_gt2) <= tol:
                            matched_any = True
                            break
                    if matched_any:
                        break
                if not matched_any:
                    fp += 1

        # finally F1
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        return 2 * (precision * recall) / max(precision + recall, 1e-4)

