import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
import argparse
from typing import Optional, Union, List


# =========================
# Utility function: compute the mean internal nearest-neighbor distance
# =========================
def mean_internal_nn_distance(xy: np.ndarray):
    """
    Given an N x 2 coordinate matrix `xy`, compute the distance from each point to its nearest neighbor (excluding itself).
    Return (mean_distance, all nearest-neighbor distances).
    """
    if xy.shape[0] < 2:
        return 0.0, np.zeros(xy.shape[0], dtype=float)

    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)  # The first result is the point itself; the second is its nearest neighbor.
    nn = dist[:, 1]
    return float(np.mean(nn)), nn


# =========================
# Automatically infer resolution from two h5ad files using obsm['spatial'] and build a nearest-neighbor mapping
# =========================
def compute_nn_mapping_from_h5ads(
    adata_s1: AnnData,
    adata_s2: AnnData,
    id_col: str = "id",
):
    """
    Use the obsm['spatial'] coordinates from two h5ad files:

      1) Read xy coordinates from the first two columns of obsm['spatial'] in adata_s1 and adata_s2.
      2) Compute the mean internal nearest-neighbor distance for S1 and S2.
      3) Treat the larger mean as low resolution and the smaller mean as high resolution.
      4) Run nearest-neighbor search from high-resolution points -> low-resolution points using cKDTree.
      5) Filter out high-resolution points when dist > 2 * d_ref_max, where d_ref_max is the maximum internal nearest-neighbor distance in the low-resolution slice.

    Returns:
      mapping_df: DataFrame with columns:
        - high_id, low_id
        - high_x, high_y
        - low_x, low_y
        - distance
        - high_index, low_index

      meta: dict containing:
        - low_res_name   : "S1" or "S2"
        - high_res_name  : "S1" or "S2"
        - d_ref_max      : float
    """
    if "spatial" not in adata_s1.obsm_keys():
        raise KeyError("adata_s1.obsm does not contain the 'spatial' key.")
    if "spatial" not in adata_s2.obsm_keys():
        raise KeyError("adata_s2.obsm does not contain the 'spatial' key.")

    xy1 = np.asarray(adata_s1.obsm["spatial"])
    xy2 = np.asarray(adata_s2.obsm["spatial"])

    if xy1.shape[1] < 2 or xy2.shape[1] < 2:
        raise ValueError("obsm['spatial'] must contain at least two coordinate columns (x, y).")

    xy1 = xy1[:, :2]
    xy2 = xy2[:, :2]

    # Remove NA/Inf coordinates.
    def clean_xy(xy):
        mask = np.isfinite(xy).all(axis=1)
        return xy[mask, :], mask

    xy1_clean, mask1 = clean_xy(xy1)
    xy2_clean, mask2 = clean_xy(xy2)

    if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
        raise ValueError("S1 or S2 has no valid coordinates left (all values are NA/Inf?).")

    print(f"S1 valid coordinate count: {xy1_clean.shape[0]} / {xy1.shape[0]}")
    print(f"S2 valid coordinate count: {xy2_clean.shape[0]} / {xy2.shape[0]}")

    # Internal nearest-neighbor mean distance (larger mean -> sparser sampling -> lower resolution)
    mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
    mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

    print(f"S1 mean internal nearest-neighbor distance: {mean_s1:.4f}")
    print(f"S2 mean internal nearest-neighbor distance: {mean_s2:.4f}")

    if mean_s1 > mean_s2:
        low_res_name = "S1"
        high_res_name = "S2"
        low_xy, low_mask = xy1_clean, mask1
        high_xy, high_mask = xy2_clean, mask2
        nn_low = nn_s1
        adata_low, adata_high = adata_s1, adata_s2
    else:
        low_res_name = "S2"
        high_res_name = "S1"
        low_xy, low_mask = xy2_clean, mask2
        high_xy, high_mask = xy1_clean, mask1
        nn_low = nn_s2
        adata_low, adata_high = adata_s2, adata_s1

    print(f"\nAutomatically determined: {low_res_name} = low resolution，{high_res_name} = high resolution")

    d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
    print(f"Maximum internal nearest-neighbor distance in the low-resolution slice, d_ref_max = {d_ref_max:.4f}")

    # Original indices for valid points.
    high_indices_all = np.where(high_mask)[0]
    low_indices_all = np.where(low_mask)[0]

    # High-resolution -> low-resolution nearest-neighbor search.
    print("\n--- High-resolution -> low-resolution nearest-neighbor search ---")
    tree = cKDTree(low_xy)
    dist, idx = tree.query(high_xy, k=1)

    # Distance-based filtering.
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

    # Build IDs.
    def get_ids(adata, id_col_name):
        if id_col_name in adata.obs.columns:
            return adata.obs[id_col_name].astype(str).to_numpy()
        return adata.obs_names.astype(str).to_numpy()

    low_ids_all = get_ids(adata_low, id_col)
    high_ids_all = get_ids(adata_high, id_col)

    mapping = pd.DataFrame(
        {
            "high_id": high_ids_all[high_idx_clean],
            "low_id": low_ids_all[low_idx_clean],
            "high_x": adata_high.obsm["spatial"][high_idx_clean, 0],
            "high_y": adata_high.obsm["spatial"][high_idx_clean, 1],
            "low_x": adata_low.obsm["spatial"][low_idx_clean, 0],
            "low_y": adata_low.obsm["spatial"][low_idx_clean, 1],
            "distance": dist_f,
            "high_index": high_idx_clean,
            "low_index": low_idx_clean,
        }
    )

    print("\nFirst rows of the mapping table:")
    print(mapping.head())

    meta = {
        "low_res_name": low_res_name,
        "high_res_name": high_res_name,
        "d_ref_max": d_ref_max,
    }
    return mapping, meta


# =========================
# Build a new h5ad from the mapping and two input h5ad files
# =========================
def build_reassigned_h5ad_from_mapping(
    mapping: pd.DataFrame,
    meta: dict,
    adata_s1: AnnData,
    adata_s2: AnnData,
    out_h5ad: str,
    id_col: str = "id",
    cluster_col: str = "cluster",
    s2_cluster_col: Union[str, List[str]] = "Manual_annotation",
    scale_by_mapping_factor: bool = True,
):
    """
    Build a new h5ad:

      - The low-resolution h5ad provides the expression matrix.
      - The new obsm['spatial'] uses high-resolution coordinates.
      - Each new point copies the expression from its corresponding low_index, with optional 1/k scaling.
      - cluster is mapped from the low-resolution h5ad when available.
      - Optionally preserve Manual_annotation from the low-resolution slice.
      - Additionally copy selected obs columns from the high-resolution slice.
        Output names follow {high}_{col} when needed (for example s2_cluster or s2_barcode_S2).
    """
    if meta["low_res_name"] == "S1":
        adata_low = adata_s1
        adata_high = adata_s2
        low_name = "S1"
        high_name = "S2"
    else:
        adata_low = adata_s2
        adata_high = adata_s1
        low_name = "S2"
        high_name = "S1"

    print(f"\nIn the reassigned h5ad: {low_name} is used as the low-resolution expression donor, and {high_name} is used as the high-resolution spatial reference.")

    if mapping.shape[0] == 0:
        raise ValueError("The mapping is empty; no matched points were found.")

    low_idx = mapping["low_index"].to_numpy(dtype=int)
    high_idx = mapping["high_index"].to_numpy(dtype=int)

    # ---- Extract the low-resolution expression matrix ----
    X_low = adata_low.X
    if sparse.isspmatrix_coo(X_low):
        print("[INFO] Low-resolution X is a coo_matrix; converting it to csr_matrix.")
        X_low = X_low.tocsr()

    if sparse.issparse(X_low):
        X_new = X_low[low_idx, :]
    else:
        X_new = np.asarray(X_low)[low_idx, :]

    # ---- 1/k scaling ----
    if scale_by_mapping_factor:
        count_map = pd.Series(low_idx).value_counts()
        mapping_factor = pd.Series(low_idx).map(count_map).to_numpy()
        if np.any(mapping_factor <= 0):
            raise ValueError("Detected mapping_factor <= 0; the mapping count is invalid.")
        scale = 1.0 / mapping_factor

        if sparse.issparse(X_new):
            X_new = X_new.multiply(scale[:, None])
            if sparse.isspmatrix_coo(X_new):
                print("[INFO] X_new is a coo_matrix; converting it to csr_matrix before writing the h5ad file.")
                X_new = X_new.tocsr()
        else:
            X_new = X_new * scale[:, None]

    # ---- Build obs ----
    n_obs_new = mapping.shape[0]
    obs_names = [f"reassign_{i}" for i in range(n_obs_new)]
    obs = pd.DataFrame(index=pd.Index(obs_names, name=None))

    obs["low_id"] = mapping["low_id"].astype(str).values
    obs["high_id"] = mapping["high_id"].astype(str).values

    # Map cluster from the low-resolution slice.
    if cluster_col in adata_low.obs.columns:
        col_src = adata_low.obs[cluster_col]
        col_src_str = col_src.astype(str)
        cats = pd.unique(col_src_str)
        mapped_cluster_str = col_src_str.iloc[low_idx].reset_index(drop=True)
        obs["cluster"] = pd.Categorical(mapped_cluster_str.values, categories=cats, ordered=False)
    else:
        print(f"[INFO] The low-resolution h5ad has no obs['{cluster_col}']; skipping cluster mapping.")

    # ---- Preserve Manual_annotation from the low-resolution slice if available ----
    low_obs_sel = adata_low.obs.iloc[low_idx].copy().reset_index(drop=True)
    if "Manual_annotation" in low_obs_sel.columns:
        obs["Manual_annotation"] = low_obs_sel["Manual_annotation"].astype(str).values
    else:
        print("[INFO] The low-resolution h5ad has no obs['Manual_annotation']; skipping Manual_annotation.")

    # ---- Additional high-resolution columns to copy (supports multiple columns; missing columns are skipped) ----
    high_obs_sel = adata_high.obs.iloc[high_idx].copy().reset_index(drop=True)

    # Normalize the argument into a list.
    if isinstance(s2_cluster_col, str):
        s2_cols = [s2_cluster_col]
    else:
        s2_cols = list(s2_cluster_col)

    written_cols = []
    # for col in s2_cols:
    #     new_col_name = f"{high_name.lower()}_{col}"
    #     if col in high_obs_sel.columns:
    #         obs[new_col_name] = high_obs_sel[col].astype(str).values
    #         written_cols.append(new_col_name)
    #     else:
    #         print(f"[WARNING] The high-resolution ({high_name}) h5ad has no obs['{col}']; skipping this column.")
    for col in s2_cols:
        if col not in high_obs_sel.columns:
            print(f"[WARNING] The high-resolution ({high_name}) h5ad has no obs['{col}']; skipping this column.")
            continue

        # Default behavior: do not add a prefix.
        out_name = col

        # If the new h5ad obs already contains the same column name, do not overwrite it; write the high-resolution column using a prefixed name instead.
        if out_name in obs.columns:
            out_name = f"{high_name.lower()}_{col}"
            print(
                f"[WARNING] obs['{col}'] already exists in the new h5ad and will not be overwritten; "
                f"the high-resolution ({high_name}) column will be written to obs['{out_name}'] instead."
            )

        obs[out_name] = high_obs_sel[col].astype(str).values


    # ---- Inherit var from the low-resolution data ----
    var = adata_low.var.copy()
    var_names = adata_low.var_names.copy()

    adata_new = AnnData(X=X_new, obs=obs, var=var)
    adata_new.var_names = var_names

    # ---- Spatial coordinates: use the high-resolution coordinates ----
    adata_new.obsm["spatial"] = mapping[["high_x", "high_y"]].to_numpy(dtype=float)

    # Distance information (not an annotation column)
    adata_new.obs["knn_dist"] = mapping["distance"].values

    # meta
    adata_new.uns["reassignment_meta"] = {
        "low_res_name": meta["low_res_name"],
        "high_res_name": meta["high_res_name"],
        "d_ref_max": float(meta["d_ref_max"]),
        "requested_high_obs_cols": s2_cols,
        "written_high_obs_cols": written_cols,
    }

    # ---- Save ----
    out_dir = os.path.dirname(out_h5ad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    adata_new.write_h5ad(out_h5ad, compression="gzip")
    print(f"\n[OK] New h5ad saved: {out_h5ad}")
    print(f"   Shape: {adata_new.n_obs} × {adata_new.n_vars}")
    print("   obs columns:", list(adata_new.obs.columns))
    print("   obsm keys:", list(adata_new.obsm.keys()))
    if "cluster" in adata_new.obs:
        print("   cluster dtype:", adata_new.obs["cluster"].dtype)
    else:
        print("   cluster dtype: N/A")

    return adata_new


# =========================
# Pipeline entry function: spomialign_reassignment
# =========================
def spomialign_reassignment(
    s1_h5ad: str,
    s2_h5ad: str,
    out_h5ad: str,
    map_csv: Optional[str] = None,
    id_col: str = "id",
    cluster_col: str = "cluster",
    s2_cluster_col: Union[str, List[str]] = "Manual_annotation",
    scale_by_mapping_factor: bool = True,
):
    """
    SPOmiAlign reassignment pipeline (pure h5ad version)
    """
    print(f"Reading S1 h5ad: {s1_h5ad}")
    adata_s1 = sc.read_h5ad(s1_h5ad)
    print(f"Reading S2 h5ad: {s2_h5ad}")
    adata_s2 = sc.read_h5ad(s2_h5ad)

    mapping, meta = compute_nn_mapping_from_h5ads(
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        id_col=id_col,
    )

    if map_csv is not None:
        out_dir = os.path.dirname(map_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mapping.to_csv(map_csv, index=False)
        print(f"\nIntermediate mapping table saved: {map_csv}")

    adata_new = build_reassigned_h5ad_from_mapping(
        mapping=mapping,
        meta=meta,
        adata_s1=adata_s1,
        adata_s2=adata_s2,
        out_h5ad=out_h5ad,
        id_col=id_col,
        cluster_col=cluster_col,
        s2_cluster_col=s2_cluster_col,
        scale_by_mapping_factor=scale_by_mapping_factor,
    )
    return adata_new


# =========================
# Command-line entry point
# =========================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "SPOmiAlign reassignment (pure h5ad version):\n"
            "Automatically determine which of S1/S2 is high resolution or low resolution, then search nearest neighbors from high-resolution points to low-resolution points, "
            "and build a new h5ad with expression inherited from the low-resolution slice.\n"
            "The new h5ad preserves low-resolution Manual_annotation when available, "
            "and can also append multiple high-resolution obs columns such as {high}_{col} (for example s2_cluster)."
        )
    )
    parser.add_argument("--s1_h5ad", "-h1", required=True, help="Path to the S1 h5ad file (must contain obsm['spatial'])")
    parser.add_argument("--s2_h5ad", "-h2", required=True, help="Path to the S2 h5ad file (must contain obsm['spatial'])")
    parser.add_argument("--out_h5ad", "-o", required=True, help="Output h5ad path")
    parser.add_argument("--map_csv", "-m", default=None, help="Optional output path for the intermediate mapping CSV")
    parser.add_argument("--id_col", default="id", help="obs column to use as the ID; obs_names are used if the column is missing")
    parser.add_argument("--cluster_col", default="cluster", help="Name of the cluster column to copy from the low-resolution h5ad")

    parser.add_argument(
        "--s2_cluster_col",
        nargs="+",
        default=["Manual_annotation"],
        help=(
            "obs column names to copy from the high-resolution slice into the new h5ad. Multiple values are allowed. "
            "Example: --s2_cluster_col cluster barcode_S2"
        ),
    )
    parser.add_argument("--no_scale", action="store_true", help="Disable 1/k scaling (skip expression-density normalization)")
    args = parser.parse_args()

    spomialign_reassignment(
        s1_h5ad=args.s1_h5ad,
        s2_h5ad=args.s2_h5ad,
        out_h5ad=args.out_h5ad,
        map_csv=args.map_csv,
        id_col=args.id_col,
        cluster_col=args.cluster_col,
        s2_cluster_col=args.s2_cluster_col,
        scale_by_mapping_factor=not args.no_scale,
    )


if __name__ == "__main__":
    main()
