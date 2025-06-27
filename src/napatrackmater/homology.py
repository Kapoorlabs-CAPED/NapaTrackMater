import numpy as np
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm 
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from scipy.spatial.distance import pdist, squareform

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



def vr_entropy_all_frames(
    df: pd.DataFrame,
    spatial_cols=('z', 'y', 'x'),
    metric='euclidean',
    homology_dims=(0, 1, 2),
    **vr_kwargs
):
    """
    Compute Vietoris Rips persistence *once* for every movie frame
    and return both the raw diagrams and a compact entropy feature.

    Parameters
    ----------
    df : DataFrame
        Must contain a 't' column and the spatial columns.
    spatial_cols : tuple[str]
        Column names in the order used to build the point cloud.
    metric : str
        Any metric accepted by gtda (default 'euclidean').
    homology_dims : tuple[int]
        Which homology dimensions to keep.
    vr_kwargs : dict
        Extra keywords forwarded to VietorisRipsPersistence, e.g.
        `max_edge_length`, `n_jobs`, etc.

    Returns
    -------
    t_sorted : (n_times,) ndarray
        Sorted, unique time stamps.
    diagrams : (n_times, n_pairs, 3) ndarray
        Persistence diagrams in birth death dim format.
    entropy : (n_times, len(homology_dims)) ndarray
        Persistence-entropy features (one vector per frame).
    """
    # --- build the 3-D array of point clouds -----------------------------
    t_sorted = np.sort(df['t'].unique())
    point_clouds = np.stack([
        df.loc[df['t'] == t, spatial_cols].to_numpy(float)
        for t in t_sorted
    ])

    # --- gtda transformers ------------------------------------------------
    vr = VietorisRipsPersistence(
        metric=metric,
        homology_dimensions=list(homology_dims),
        **vr_kwargs
    )
    pe = PersistenceEntropy()

    diagrams = vr.fit_transform(point_clouds)     # shape (n_times, n_pairs, 3)
    entropy  = pe.fit_transform(diagrams)         # shape (n_times, |dims|)

    return t_sorted, diagrams, entropy    


def vr_entropy_generator(
    df: pd.DataFrame,
    spatial_cols=('z', 'y', 'x'),
    metric='euclidean',
    homology_dims=(0, 1, 2),
    **vr_kwargs
):
    """
    Yield a `(t, diagrams, entropy_vec)` triple for *each* frame.
    Useful when the whole movie doesn’t fit in RAM.

    Examples
    --------
    betti_0 = []
    for t, diag, ent in vr_entropy_generator(tracks_dataframe):
        betti_0.append((t, (diag[0][:,1] > diag[0][:,0]).sum()))   # β₀ count
    """
    vr = VietorisRipsPersistence(
        metric=metric,
        homology_dimensions=list(homology_dims),
        **vr_kwargs
    )
    pe = PersistenceEntropy()

    for t in np.sort(df['t'].unique()):
        pts = df.loc[df['t'] == t, spatial_cols].to_numpy(float)
        if pts.size == 0:
            continue
        diag      = vr.fit_transform(pts[None, :, :])[0]  # add fake batch axis
        entropy_v = pe.fit_transform(diag[None, :, :])[0]
        yield t, diag, entropy_v    