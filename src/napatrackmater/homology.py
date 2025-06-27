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
    title=None,
    sort_by_persistence=True,
):
    """
    Visualise a barcode over time: one bar per feature, showing its birth and death scale.

    Parameters
    ----------
    diagrams_by_time : dict[int, list[np.ndarray]]
        Map of time -> list of diagrams (one per dimension).
    dim : int
        Homology dimension (e.g. 1 for loops).
    sort_by_persistence : bool
        If True, sort bars by persistence (death - birth).
    max_bars : int
        Max number of bars to show (to avoid overload).
    """
    all_bars = []

    for t, dgms in diagrams_by_time.items():
        if len(dgms) <= dim:
            continue
        for birth, death in dgms[dim]:
            all_bars.append((t, birth, death))

    if not all_bars:
        print("No features found.")
        return

    all_bars = np.array(all_bars)
    persistence = all_bars[:, 2] - all_bars[:, 1]

    # optionally sort by persistence
    if sort_by_persistence:
        sorted_idx = np.argsort(-persistence)
        all_bars = all_bars[sorted_idx]


    # plot each bar as a horizontal line
    plt.figure(figsize=(10, 6))
    for i, (t, b, d) in enumerate(all_bars):
        plt.hlines(y=i, xmin=b, xmax=d, color='royalblue', linewidth=1.5)
        plt.plot([b, d], [i, i], 'o', color='black', markersize=2)

    plt.xlabel("Scale")
    plt.ylabel("Topological feature (sorted)")
    plt.title(title or f"Barcode plot over time (H{dim})")
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


