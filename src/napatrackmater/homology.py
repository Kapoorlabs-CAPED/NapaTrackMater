import numpy as np
import seaborn as sns
from pathlib import Path
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm 
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def save_barcodes_and_stats(
    diagrams_by_time: dict,
    dims=(0, 1),
    save_dir="barcodes_per_frame",
    max_bars=None,
    plot_joint_hist=True,
    csv_loop_stats=True,
):
    """
    For every time-point save:
      • barcode plot PNG (H_dim for dim in dims)
      • optional: combined histogram/KDE of H1 persistence across time
      • optional: CSV per timepoint with birth/death/persistence (H1)
    """
    save_dir = Path(save_dir)
    png_dir = save_dir / "png"
    csv_dir = save_dir / "csv"
    hist_path = save_dir / "combined_histogram_H1_persistence.png"

    png_dir.mkdir(parents=True, exist_ok=True)
    if csv_loop_stats:
        csv_dir.mkdir(exist_ok=True)

    all_persistence = []

    for t, dgms in diagrams_by_time.items():
        # ---------- BARCODE PLOT --------------------------------------
        fig, axs = plt.subplots(
            len(dims), 1, figsize=(8, 2.5 * len(dims)), squeeze=False
        )
        for i, dim in enumerate(dims):
            ax = axs[i, 0]
            if dim >= len(dgms) or dgms[dim].size == 0:
                ax.set_axis_off()
                continue

            diag = dgms[dim]
            if max_bars is not None and len(diag) > max_bars:
                pers = diag[:, 1] - diag[:, 0]
                idx = np.argsort(-pers)[:max_bars]
                diag = diag[idx]

            for j, (b, d) in enumerate(diag):
                ax.hlines(j, b, d, color="tab:blue")
                ax.plot([b, d], [j, j], "k.", ms=2)
            ax.set_title(f"H{dim} barcode at t={t}")
            ax.set_xlabel("Filtration scale")
            ax.set_ylabel("Feature index")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_dir / f"barcode_t{int(t):04d}.png", dpi=200)
        plt.close(fig)

        # ---------- COLLECT H1 persistences ---------------------------
        if plot_joint_hist and len(dgms) > 1 and dgms[1].size:
            persistence = dgms[1][:, 1] - dgms[1][:, 0]
            all_persistence.append(
                pd.DataFrame(
                    {"persistence": persistence, "time": t}
                )
            )

        # ---------- CSV export for H1 --------------------------------
        if csv_loop_stats and len(dgms) > 1 and dgms[1].size:
            df_out = pd.DataFrame(dgms[1], columns=["birth", "death"])
            df_out["persistence"] = df_out["death"] - df_out["birth"]
            df_out.to_csv(csv_dir / f"loops_t{int(t):04d}.csv", index=False)

    # ---------- PLOT OVERLAID HISTOGRAM (SEABORN) ---------------------
    if plot_joint_hist and all_persistence:
        full_df = pd.concat(all_persistence, ignore_index=True)

        # ----- assign a bin id to every time value ----------------
        full_df["bin"] = (full_df["time"] // 50).astype(int)    

        # palette with as many distinct colours as bins
        n_bins = full_df["bin"].nunique()
        palette = sns.color_palette("tab10", n_bins)  

        plt.figure(figsize=(10, 6))

        for bin_id, colour in zip(sorted(full_df["bin"].unique()), palette):
            times_in_bin = full_df.loc[full_df["bin"] == bin_id, "time"].unique()
            for t in times_in_bin:
                subset = full_df[(full_df["time"] == t)]
                label = f"{bin_id*50}-{bin_id*50+49}" if t == times_in_bin[0] else None
                sns.kdeplot(
                    subset["persistence"],
                    color=colour,
                    label=label,
                    linewidth=1.4,
                    alpha=0.40,
                )

        plt.title("H1 persistence KDEs (grouped in 50-frame bins)")
        plt.xlabel("Persistence (death - birth)")
        plt.ylabel("Density")
        plt.legend(title="Time bin (t)")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()



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


