import numpy as np
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
    dim=1,
    save_path=None,
    title=None
):
    """
    Plot (birth, death) pairs across time, for features in H_dim.

    Parameters
    ----------
    diagrams_by_time : dict[int, list[np.ndarray]]
        Maps each t to its list of persistence diagrams.
    dim : int
        Homology dimension to show (0 = components, 1 = loops, 2 = voids).
    save_path : str or None
        If set, saves plot to this file.
    """
    all_births, all_deaths, all_times = [], [], []

    for t, dgms in diagrams_by_time.items():
        diag = dgms[dim]
        if diag.size == 0:
            continue
        for bd in diag:
            birth, death = bd
            all_births.append(birth)
            all_deaths.append(death)
            all_times.append(t)

    plt.figure(figsize=(6, 4))
    plt.scatter(all_times, all_births, s=10, label="birth", alpha=0.5)
    plt.scatter(all_times, all_deaths, s=10, label="death", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Scale (birth/death)")
    plt.title(title or f"Persistence (H{dim}) over time")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


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


