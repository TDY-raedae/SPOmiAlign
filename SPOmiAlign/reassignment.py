#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize, LinearSegmentedColormap

warnings.filterwarnings("ignore")


# =========================
# colormaps
# =========================
# S1 -> orange
# S2 -> blue
paper_blue = ["#6baed6", "#4292c6", "#2171b5", "#08519c", "#084594", "#08306b"]
paper_orange = ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#e6550d", "#a63603", "#7f2704"]

cmap_blue = LinearSegmentedColormap.from_list("paper_blue_deeper", paper_blue)
cmap_orange = LinearSegmentedColormap.from_list("paper_orange_deeper", paper_orange)


# =========================
# basic utils
# =========================
def get_spatial_from_adata(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Read 2D spatial coordinates from adata.obsm[spatial_key].
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm does not contain '{spatial_key}'。available keys：{list(adata.obsm.keys())}"
        )

    xy = np.asarray(adata.obsm[spatial_key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] has invalid shape：{xy.shape}，expected at least (n_spots, 2)"
        )

    return xy[:, :2].astype(float)


def mean_internal_nn_distance(xy: np.ndarray):
    """
    Compute mean internal nearest-neighbor distance as a rough resolution estimate.
    """
    if xy.shape[0] < 2:
        return 0.0, np.zeros(xy.shape[0], dtype=float)

    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)
    nn = dist[:, 1]
    return float(np.mean(nn)), nn


def robust_norm(values, q_low=5, q_high=99):
    lo = np.percentile(values, q_low)
    hi = np.percentile(values, q_high)
    if hi <= lo:
        hi = lo + 1e-6
    return Normalize(vmin=lo, vmax=hi)


def get_umi_values(adata: AnnData):
    """
    Prefer obs['umi'] or obs['total_counts']; fall back to X.sum(axis=1).
    """
    if "umi" in adata.obs.columns:
        return np.asarray(adata.obs["umi"]).astype(float), "obs['umi']"
    if "total_counts" in adata.obs.columns:
        return np.asarray(adata.obs["total_counts"]).astype(float), "obs['total_counts']"

    X = adata.X
    if sparse.issparse(X):
        vals = np.asarray(X.sum(axis=1)).ravel().astype(float)
    else:
        vals = np.asarray(X).sum(axis=1).astype(float)

    return vals, "X.sum(axis=1)"


def filter_valid_coords_values(coords: np.ndarray, values: np.ndarray):
    mask = (
        np.isfinite(coords[:, 0]) &
        np.isfinite(coords[:, 1]) &
        np.isfinite(values)
    )
    return coords[mask], values[mask]


def print_plot_info(name, coords, values, value_src):
    print(f"\n===== {name} =====")
    print(f"value source: {value_src}")
    print(f"n spots: {len(values)}")
    print(f"x range: {coords[:, 0].min():.3f} ~ {coords[:, 0].max():.3f}")
    print(f"y range: {coords[:, 1].min():.3f} ~ {coords[:, 1].max():.3f}")
    print(f"value range: {values.min():.3f} ~ {values.max():.3f}")
    print(
        f"value q1/q50/q99: "
        f"{np.percentile(values,1):.3f}, "
        f"{np.percentile(values,50):.3f}, "
        f"{np.percentile(values,99):.3f}"
    )


def estimate_auto_spot_size(
    xy: np.ndarray,
    fig_size=(8, 8),
    fill_ratio: float = 0.94,
    percentile: float = 10,
    min_size: float = 6,
    max_size: float = 900,
):
    """
    Automatically estimate scatter spot_size (parameter s) from spatial coordinates.
    """
    if xy.shape[0] < 2:
        return min_size

    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)
    nn = dist[:, 1]
    nn = nn[np.isfinite(nn)]

    if nn.size == 0:
        return min_size

    d_typical = np.percentile(nn, percentile)
    if d_typical <= 0:
        return min_size

    x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
    y_min, y_max = np.min(xy[:, 1]), np.max(xy[:, 1])

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    fig_w, fig_h = fig_size
    ax_w_pt = fig_w * 72.0
    ax_h_pt = fig_h * 72.0

    dx_pt = d_typical / x_range * ax_w_pt
    dy_pt = d_typical / y_range * ax_h_pt

    side_pt = min(dx_pt, dy_pt) * fill_ratio
    s = side_pt ** 2

    s = float(np.clip(s, min_size, max_size))
    return s


def auto_make_out_h5ad_path(
    out_dir: str,
    reassignment_direction: str,
    meta: dict,
    s1_h5ad: str,
    s2_h5ad: str,
) -> str:
    """
    Automatically generate output h5ad path:
    - high_to_low: use the high-resolution h5ad filename with reassigned_ prefix
    - low_to_high: use the low-resolution h5ad filename with reassigned_ prefix
    """
    os.makedirs(out_dir, exist_ok=True)

    low_res_name = meta["low_res_name"]
    high_res_name = meta["high_res_name"]

    if reassignment_direction == "high_to_low":
        ref_name = high_res_name
    else:
        ref_name = low_res_name

    ref_h5ad = s1_h5ad if ref_name == "S1" else s2_h5ad
    base = os.path.basename(ref_h5ad)
    out_name = f"reassigned_{base}"
    return os.path.join(out_dir, out_name)


def aggregate_strings_by_target(src_values, tgt_idx, n_target):
    """
    Aggregate string labels onto target spots.
    Returns:
    1) mode_values: mode label for each target spot; NA if none
    2) all_values: concatenated labels for each target spot
    """
    groups = [[] for _ in range(n_target)]
    for v, t in zip(src_values, tgt_idx):
        groups[t].append(str(v))

    mode_values = []
    all_values = []

    for arr in groups:
        if len(arr) == 0:
            mode_values.append("NA")
            all_values.append("")
        else:
            ser = pd.Series(arr, dtype="object")
            vc = ser.value_counts()
            mode_values.append(str(vc.index[0]))
            all_values.append("|".join(arr))

    return np.asarray(mode_values, dtype=object), np.asarray(all_values, dtype=object)


# =========================
# plotting
# =========================
def plot_h5ad_umi_squares(
    adata: AnnData,
    out_png: str,
    title: str = "",
    spot_size: float = None,
    spatial_key: str = "spatial",
    cmap=None,
    fig_size=(8, 8),
    auto_fill_ratio: float = 0.88,
):
    """
    Plot each spot as a square colored by UMI.
    """
    xy = get_spatial_from_adata(adata, spatial_key)
    umi, value_src = get_umi_values(adata)

    xy, umi = filter_valid_coords_values(xy, umi)

    if umi.size == 0:
        raise ValueError("No valid spots are available for plotting.")

    print_plot_info(title if title else "plot", xy, umi, value_src)

    if cmap is None:
        cmap = cmap_blue

    norm = robust_norm(umi, q_low=5, q_high=99)

    if spot_size is None:
        spot_size = estimate_auto_spot_size(
            xy,
            fig_size=fig_size,
            fill_ratio=auto_fill_ratio,
            percentile=10,
            min_size=6,
            max_size=900,
        )
        print(f"Auto-estimated spot_size = {spot_size:.2f}")
    else:
        print(f"Using manual spot_size = {spot_size:.2f}")

    fig, ax = plt.subplots(figsize=fig_size)
    sca = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=umi,
        cmap=cmap,
        norm=norm,
        s=spot_size,
        marker="s",
        linewidths=0,
        alpha=1.0
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=16)

    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("UMI", fontsize=12)

    plt.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] UMI visualization saved: {out_png}")


# =========================
# mapping
# =========================
def compute_nn_mapping_from_h5ads(
    adata_s1: AnnData,
    adata_s2: AnnData,
    reassignment_direction: str = "low_to_high",
    s1_spatial_key: str = "spatial",
    s2_spatial_key: str = "spatial",
):
    """
    Build mapping from two h5ad files using selected coordinate keys and auto-detected resolution.
    """
    if reassignment_direction not in {"low_to_high", "high_to_low"}:
        raise ValueError("reassignment_direction must be 'low_to_high' or 'high_to_low'.")

    xy1 = get_spatial_from_adata(adata_s1, s1_spatial_key)
    xy2 = get_spatial_from_adata(adata_s2, s2_spatial_key)

    def clean_xy(xy):
        mask = np.isfinite(xy).all(axis=1)
        return xy[mask, :], mask

    xy1_clean, mask1 = clean_xy(xy1)
    xy2_clean, mask2 = clean_xy(xy2)

    if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
        raise ValueError("S1 or S2 has no valid coordinates left (all values are NA/Inf?).")

    print(f"S1 valid coordinate count: {xy1_clean.shape[0]} / {xy1.shape[0]}")
    print(f"S2 valid coordinate count: {xy2_clean.shape[0]} / {xy2.shape[0]}")

    mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
    mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

    print(f"S1 mean internal nearest-neighbor distance: {mean_s1:.4f}")
    print(f"S2 mean internal nearest-neighbor distance: {mean_s2:.4f}")

    # Larger mean nearest-neighbor distance means sparser and lower-resolution.
    if mean_s1 > mean_s2:
        low_res_name = "S1"
        high_res_name = "S2"
        low_xy, low_mask = xy1_clean, mask1
        high_xy, high_mask = xy2_clean, mask2
        nn_low = nn_s1
    else:
        low_res_name = "S2"
        high_res_name = "S1"
        low_xy, low_mask = xy2_clean, mask2
        high_xy, high_mask = xy1_clean, mask1
        nn_low = nn_s2

    print(f"\nAutomatically determined: {low_res_name} = low resolution，{high_res_name} = high resolution")

    d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
    print(f"Maximum internal nearest-neighbor distance in the low-resolution slice, d_ref_max = {d_ref_max:.4f}")

    low_indices_all = np.where(low_mask)[0]
    high_indices_all = np.where(high_mask)[0]

    tree_low = cKDTree(low_xy)
    dist, idx = tree_low.query(high_xy, k=1)

    if d_ref_max > 0:
        valid = dist <= 2.0 * d_ref_max
    else:
        valid = np.ones_like(dist, dtype=bool)

    n_drop = int(np.sum(~valid))
    print(f"Distance filtering: removed {n_drop} high-resolution points (dist > 2 * d_ref_max).")

    dist_f = dist[valid]
    idx_f = idx[valid]

    high_idx_clean = high_indices_all[valid]
    low_idx_clean = low_indices_all[idx_f]

    if reassignment_direction == "low_to_high":
        mapping = pd.DataFrame(
            {
                "source_index": low_idx_clean,
                "target_index": high_idx_clean,
                "source_x": low_xy[idx_f][:, 0],
                "source_y": low_xy[idx_f][:, 1],
                "target_x": high_xy[valid][:, 0],
                "target_y": high_xy[valid][:, 1],
                "distance": dist_f,
            }
        )
        print("\nCurrent mode: low_to_high (low-resolution expression -> high-resolution coordinates)")
    else:
        mapping = pd.DataFrame(
            {
                "source_index": high_idx_clean,
                "target_index": low_idx_clean,
                "source_x": high_xy[valid][:, 0],
                "source_y": high_xy[valid][:, 1],
                "target_x": low_xy[idx_f][:, 0],
                "target_y": low_xy[idx_f][:, 1],
                "distance": dist_f,
            }
        )
        print("\nCurrent mode: high_to_low (sum high-resolution expression -> low-resolution coordinates)")

    print("\nFirst rows of the mapping table:")
    print(mapping.head())

    meta = {
        "low_res_name": low_res_name,
        "high_res_name": high_res_name,
        "d_ref_max": d_ref_max,
        "reassignment_direction": reassignment_direction,
        "s1_spatial_key": s1_spatial_key,
        "s2_spatial_key": s2_spatial_key,
    }
    return mapping, meta


# =========================
# build reassigned h5ad
# =========================
def build_reassigned_h5ad_from_mapping(
    mapping: pd.DataFrame,
    meta: dict,
    adata_s1: AnnData,
    adata_s2: AnnData,
    out_h5ad: str,
    scale_by_mapping_factor: bool = True,
    reserved_col: str = None,
):
    """
    Build a new h5ad from mapping and two h5ad files.

    reserved_col logic:
    - low_to_high: new h5ad keeps high by default, so reserved_col is kept from low
    - high_to_low: new h5ad keeps low by default, so reserved_col is kept from high
    """
    low_res_name = meta["low_res_name"]
    high_res_name = meta["high_res_name"]
    reassignment_direction = meta["reassignment_direction"]

    adata_low = adata_s1 if low_res_name == "S1" else adata_s2
    adata_high = adata_s2 if high_res_name == "S2" else adata_s1

    low_spatial_key = meta["s1_spatial_key"] if low_res_name == "S1" else meta["s2_spatial_key"]
    high_spatial_key = meta["s1_spatial_key"] if high_res_name == "S1" else meta["s2_spatial_key"]

    if reassignment_direction == "low_to_high":
        print("\nBuilding new low_to_high h5ad ...")

        src_idx = mapping["source_index"].to_numpy(dtype=int)   # low
        tgt_idx = mapping["target_index"].to_numpy(dtype=int)   # high

        X_low = adata_low.X
        if sparse.issparse(X_low):
            X_low = X_low.tocsr()
        else:
            X_low = np.asarray(X_low)

        counts = pd.Series(src_idx).value_counts().to_dict()

        if sparse.issparse(adata_low.X):
            rows = []
            for s in src_idx:
                row = X_low.getrow(s).copy()
                if scale_by_mapping_factor:
                    k = counts.get(s, 1)
                    if k > 1:
                        row.data = row.data / float(k)
                rows.append(row)
            X_new = sparse.vstack(rows, format="csr")
        else:
            rows = []
            for s in src_idx:
                row = X_low[s].copy()
                if scale_by_mapping_factor:
                    k = counts.get(s, 1)
                    row = row / float(k)
                rows.append(row)
            X_new = np.vstack(rows)

        # New coordinates are high-resolution, so obs mainly comes from high.
        obs_new = adata_high.obs.iloc[tgt_idx].copy()
        obs_new["reassigned_from"] = src_idx
        obs_new["reassigned_to"] = tgt_idx

        # reserved_col in low_to_high should keep columns from low
        if reserved_col is not None:
            if reserved_col not in adata_low.obs.columns:
                raise KeyError(
                    f"reserved_col='{reserved_col}' is not in low-slice obs. "
                    f"Available columns: {list(adata_low.obs.columns)}"
                )
            obs_new[f"reserved_low_{reserved_col}"] = (
                adata_low.obs.iloc[src_idx][reserved_col].astype(str).to_numpy()
            )

        var_new = adata_low.var.copy()
        spatial_new = get_spatial_from_adata(adata_high, high_spatial_key)[tgt_idx]

        adata_new = AnnData(X=X_new, obs=obs_new, var=var_new)
        adata_new.obsm["spatial"] = spatial_new

    else:
        print("\nBuilding new high_to_low h5ad ...")

        src_idx = mapping["source_index"].to_numpy(dtype=int)   # high
        tgt_idx = mapping["target_index"].to_numpy(dtype=int)   # low

        X_high = adata_high.X
        if sparse.issparse(X_high):
            X_high = X_high.tocsr()
        else:
            X_high = np.asarray(X_high)

        n_low = adata_low.n_obs
        n_genes = adata_high.n_vars

        mapped_counts = np.bincount(tgt_idx, minlength=n_low)

        if sparse.issparse(adata_high.X):
            data_list = []
            row_list = []
            col_list = []

            for s, t in zip(src_idx, tgt_idx):
                row = X_high.getrow(s)
                if row.nnz == 0:
                    continue
                coo = row.tocoo()
                data_list.append(coo.data)
                row_list.append(np.full(coo.col.shape, t, dtype=np.int64))
                col_list.append(coo.col.astype(np.int64))

            if len(data_list) > 0:
                data = np.concatenate(data_list)
                rows = np.concatenate(row_list)
                cols = np.concatenate(col_list)

                X_new = sparse.coo_matrix(
                    (data, (rows, cols)),
                    shape=(n_low, n_genes)
                ).tocsr()
                X_new.sum_duplicates()
            else:
                X_new = sparse.csr_matrix((n_low, n_genes), dtype=X_high.dtype)

        else:
            X_new = np.zeros((n_low, n_genes), dtype=X_high.dtype)
            for s, t in zip(src_idx, tgt_idx):
                X_new[t] += X_high[s]

        # New coordinates are low-resolution, so obs mainly comes from low.
        obs_new = adata_low.obs.copy()
        obs_new["mapped_high_count"] = mapped_counts.astype(int)

        # reserved_col in high_to_low should keep columns from high
        if reserved_col is not None:
            if reserved_col not in adata_high.obs.columns:
                raise KeyError(
                    f"reserved_col='{reserved_col}' is not in high-slice obs. "
                    f"Available columns: {list(adata_high.obs.columns)}"
                )
            high_vals = adata_high.obs.iloc[src_idx][reserved_col].astype(str).to_numpy()
            mode_vals, all_vals = aggregate_strings_by_target(high_vals, tgt_idx, n_low)
            obs_new[f"reserved_high_{reserved_col}"] = mode_vals
            obs_new[f"reserved_high_{reserved_col}_all"] = all_vals

        var_new = adata_high.var.copy()
        spatial_new = get_spatial_from_adata(adata_low, low_spatial_key)

        adata_new = AnnData(X=X_new, obs=obs_new, var=var_new)
        adata_new.obsm["spatial"] = spatial_new

    adata_new.uns["reassignment_meta"] = dict(meta)
    adata_new.uns["reassignment_meta"]["scale_by_mapping_factor"] = bool(scale_by_mapping_factor)
    adata_new.uns["reassignment_meta"]["reserved_col"] = reserved_col
    adata_new.uns["reassignment_meta"]["reserved_rule"] = (
        "low_to_high -> reserved from low; high_to_low -> reserved from high"
    )

    if sparse.issparse(adata_new.X):
        adata_new.X = adata_new.X.tocsr()

    print("X type before writing:", type(adata_new.X))

    out_dir = os.path.dirname(out_h5ad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    adata_new.write_h5ad(out_h5ad, compression="gzip")
    print(f"[OK] New h5ad saved: {out_h5ad}")

    return adata_new


# =========================
# main pipeline
# =========================
def spomialign_reassignment(
    s1_h5ad: str,
    s2_h5ad: str,
    out_dir: str,
    map_csv: str = None,
    s1_spatial_key: str = "spatial",
    s2_spatial_key: str = "spatial",
    scale_by_mapping_factor: bool = True,
    reassignment_direction: str = "low_to_high",
    save_plots: bool = False,
    plot_dir: str = None,
    plot_spot_size: float = None,
    reserved_col: str = None,
):
    print("Reading h5ad ...")
    adata_s1 = sc.read_h5ad(s1_h5ad)
    adata_s2 = sc.read_h5ad(s2_h5ad)

    if save_plots:
        if plot_dir is None:
            plot_dir = os.path.join(out_dir, "reassignment_plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_h5ad_umi_squares(
            adata_s1,
            out_png=os.path.join(plot_dir, "S1_umi.png"),
            title=f"S1 UMI ({s1_spatial_key})",
            spot_size=plot_spot_size,
            spatial_key=s1_spatial_key,
            cmap=cmap_orange,
        )
        plot_h5ad_umi_squares(
            adata_s2,
            out_png=os.path.join(plot_dir, "S2_umi.png"),
            title=f"S2 UMI ({s2_spatial_key})",
            spot_size=plot_spot_size,
            spatial_key=s2_spatial_key,
            cmap=cmap_blue,
        )

    mapping, meta = compute_nn_mapping_from_h5ads(
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        reassignment_direction=reassignment_direction,
        s1_spatial_key=s1_spatial_key,
        s2_spatial_key=s2_spatial_key,
    )

    out_h5ad = auto_make_out_h5ad_path(
        out_dir=out_dir,
        reassignment_direction=reassignment_direction,
        meta=meta,
        s1_h5ad=s1_h5ad,
        s2_h5ad=s2_h5ad,
    )
    print(f"\nAuto output h5ad path: {out_h5ad}")

    if map_csv is not None:
        out_dir_csv = os.path.dirname(map_csv)
        if out_dir_csv:
            os.makedirs(out_dir_csv, exist_ok=True)
        mapping.to_csv(map_csv, index=False)
        print(f"\nIntermediate mapping table saved: {map_csv}")

    adata_new = build_reassigned_h5ad_from_mapping(
        mapping=mapping,
        meta=meta,
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        out_h5ad=out_h5ad,
        scale_by_mapping_factor=scale_by_mapping_factor,
        reserved_col=reserved_col,
    )

    if save_plots:
        low_res_name = meta["low_res_name"]
        high_res_name = meta["high_res_name"]

        low_cmap = cmap_orange if low_res_name == "S1" else cmap_blue
        high_cmap = cmap_orange if high_res_name == "S1" else cmap_blue

        if reassignment_direction == "high_to_low":
            reassigned_cmap = high_cmap
        else:
            reassigned_cmap = low_cmap

        plot_h5ad_umi_squares(
            adata_new,
            out_png=os.path.join(plot_dir, f"reassigned_{reassignment_direction}_umi.png"),
            title=f"Reassigned ({reassignment_direction}) UMI",
            spot_size=plot_spot_size,
            spatial_key="spatial",
            cmap=reassigned_cmap,
        )

    return adata_new


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "SPOmiAlign reassignment (h5ad-only version):\n"
            "Automatically detects which of S1/S2 is high/low resolution and supports two directions:\n"
            "1) low_to_high: low-resolution expression assigned to high-resolution coordinates (1/k scaling)\n"
            "2) high_to_low: high-resolution expression aggregated to low-resolution coordinates (keeps all low spots; unmatched spots are zero-filled)\n"
            "Supports separate coordinate keys for S1/S2.\n"
            "Output h5ad filename is generated automatically; only out_dir is required.\n"
            "reserved_col rule: low_to_high keeps from low; high_to_low keeps from high."
        )
    )
    parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad path")
    parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad path")
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory; the new h5ad is named automatically")
    parser.add_argument("--map_csv", "-m", default=None, help="Intermediate mapping-table CSV output path (optional)")

    parser.add_argument(
        "--s1_spatial_key",
        default="spatial",
        help="Coordinate key used by S1, e.g. spatial / spatial_raw / spatial_spomialign",
    )
    parser.add_argument(
        "--s2_spatial_key",
        default="spatial",
        help="Coordinate key used by S2, e.g. spatial / spatial_raw / spatial_spomialign",
    )

    parser.add_argument("--no_scale", action="store_true", help="Disable 1/k scaling (only affects low_to_high)")

    parser.add_argument(
        "--reassignment_direction",
        choices=["low_to_high", "high_to_low"],
        default="low_to_high",
        help="Reassignment direction: low_to_high or high_to_low",
    )

    parser.add_argument(
        "--reserved_col",
        default=None,
        help="Extra obs column to keep from the other slice, e.g. Manual_annotation; omitted by default",
    )

    parser.add_argument("--save_plots", action="store_true", help="Save UMI visualizations for input/output h5ad files")
    parser.add_argument("--plot_dir", default=None, help="Plot output directory (default: reassignment_plots under out_dir)")
    parser.add_argument(
        "--plot_spot_size",
        type=float,
        default=None,
        help="Spot size for plotting (scatter s). If omitted, estimated automatically from resolution.",
    )

    args = parser.parse_args()

    spomialign_reassignment(
        s1_h5ad=args.s1_h5ad,
        s2_h5ad=args.s2_h5ad,
        out_dir=args.out_dir,
        map_csv=args.map_csv,
        s1_spatial_key=args.s1_spatial_key,
        s2_spatial_key=args.s2_spatial_key,
        scale_by_mapping_factor=(not args.no_scale),
        reassignment_direction=args.reassignment_direction,
        save_plots=args.save_plots,
        plot_dir=args.plot_dir,
        plot_spot_size=args.plot_spot_size,
        reserved_col=args.reserved_col,
    )


if __name__ == "__main__":
    main()