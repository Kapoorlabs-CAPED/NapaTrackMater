import numpy as np
import os
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm 
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_persistence_time_series(
    
    diagrams_by_time,
    dims=(0, 1),
    save_dir="barcodes_per_frame",
    max_bars=None,
):
    """
    Generate one barcode plot per timepoint and save as PNG.

    Parameters
    ----------
    diagrams_by_time : dict[int, list[np.ndarray]]
        t → diagrams list per homology dimension
    dims : tuple[int]
        Which homology dimensions to plot
    save_dir : str
        Folder to save PNGs
    max_bars : int or None
        Max number of features per plot (to reduce clutter)
    """
    os.makedirs(save_dir, exist_ok=True)

    for t, dgms in diagrams_by_time.items():
        fig, axs = plt.subplots(len(dims), 1, figsize=(8, 2.5 * len(dims)), squeeze=False)

        for i, dim in enumerate(dims):
            ax = axs[i, 0]
            if dim >= len(dgms):
                ax.axis('off')
                continue

            diag = dgms[dim]
            if diag.size == 0:
                ax.set_title(f"H{dim} (empty)")
                ax.axis("off")
                continue

            # optionally trim number of bars
            if max_bars is not None and len(diag) > max_bars:
                pers = diag[:, 1] - diag[:, 0]
                top_idx = np.argsort(-pers)[:max_bars]
                diag = diag[top_idx]

            for j, (b, d) in enumerate(diag):
                ax.hlines(y=j, xmin=b, xmax=d, color="tab:blue")
                ax.plot([b, d], [j, j], 'o', color='black', markersize=2)

            ax.set_title(f"H{dim} barcode at t={t}")
            ax.set_xlabel("Filtration scale")
            ax.set_ylabel("Feature index")
            ax.grid(True)

        fig.tight_layout()
        save_path = os.path.join(save_dir, f"barcode_t{int(t):04d}.png")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)


def vietoris_rips_at_t(
    df: pd.DataFrame,
    t,
    spatial_cols=('z', 'y', 'x'),
    max_dim=2,
    max_edge=None,
    metric='euclidean',
    normalise=False,
    plot=False,
    use_explicit_distance=True,
):
    """
    Robust VR persistent homology for a single frame.

    Parameters
    ----------
    df : DataFrame with 't' and coordinate columns
    t  : time stamp
    use_explicit_distance : if True, pre-compute a distance matrix and
                            call ripser(..., distance_matrix=True)

    Returns
    -------
    dgms : list of arrays, diagrams[dim] = [[birth, death], …]
    """
    # ------------------------------------------------------------------ #
    # 1. slice and sanitise the point cloud
    # ------------------------------------------------------------------ #
    pts = (
        df.loc[df['t'] == t, spatial_cols]
        .dropna()                     # remove any row with NaN
        .astype(float)                # force numeric, errors -> ValueError
        .to_numpy(dtype=np.float64)   # ensure contiguous float64 array
    )

    if pts.size == 0:
        raise ValueError(f"No valid points at t = {t}")

    if normalise:
        # guard against zero std which would give NaNs
        std = pts.std(0, ddof=0)
        std[std == 0] = 1.0
        pts = (pts - pts.mean(0)) / std

    # ------------------------------------------------------------------ #
    # 2. Ripser
    # ------------------------------------------------------------------ #
    if use_explicit_distance:
        dists = squareform(pdist(pts, metric=metric)).astype(np.float64)

        if np.isnan(dists).any():
            raise ValueError(f"Distance matrix for t={t} contains NaN values")

        dgms = ripser(
            dists,
            distance_matrix=True,
            maxdim=max_dim,
            thresh=max_edge if max_edge is not None else np.inf,
        )["dgms"]
    else:
        dgms = ripser(
            pts,
            maxdim=max_dim,
            thresh=max_edge,
            metric=metric,
        )["dgms"]

    # optional quick look
    if plot:
        plot_diagrams(dgms, show=True, title=f"t = {t}")

    return dgms


def diagrams_over_time(df, time_col='t', **vr_kwargs):
    unique_times = np.sort(df[time_col].unique())
    diags = {}
    for t in tqdm(unique_times, desc="VR per frame"):
        diags[t] = vietoris_rips_at_t(df, t, plot=False, **vr_kwargs)
    return diags


