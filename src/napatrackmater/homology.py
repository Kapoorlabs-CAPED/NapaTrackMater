import numpy as np
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm 
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy

def vietoris_rips_at_t(df, t,
                       spatial_cols=('z', 'y', 'x'),
                       max_dim=2,
                       max_edge=None,        
                       metric='euclidean',
                       normalise=False,
                       plot=True):
    """
    Compute VR persistent homology of points at time t.

    Parameters
    ----------
    df : pd.DataFrame with at least ['t', *spatial_cols]
    t  : scalar time stamp
    spatial_cols : iterable of coordinate columns to use
    max_dim : largest homology dimension to compute (0 -> components, 1 -> loops, …)
    max_edge : optional distance cut-off to save RAM/CPU
    metric : any metric accepted by scipy.spatial
    normalise : z-score the point cloud before TDA
    plot : whether to draw the persistence diagram

    Returns
    -------
    dgms : list of numpy arrays, one per dimension
    """
    pts = df.loc[df['t'] == t, spatial_cols].to_numpy(float)
    if pts.shape[0] == 0:
        raise ValueError(f"No points found at t = {t}")

    if normalise:
        pts = (pts - pts.mean(0)) / pts.std(0)

    dgms = ripser(
        pts,
        maxdim=max_dim,
        thresh=max_edge,
        metric=metric
    )['dgms']

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