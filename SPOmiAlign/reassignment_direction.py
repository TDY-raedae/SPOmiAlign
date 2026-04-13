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
    从 adata.obsm[spatial_key] 读取二维空间坐标
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm 中不存在 '{spatial_key}'。可用键有：{list(adata.obsm.keys())}"
        )

    xy = np.asarray(adata.obsm[spatial_key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] 的形状异常：{xy.shape}，期望至少为 (n_spots, 2)"
        )

    return xy[:, :2].astype(float)


def mean_internal_nn_distance(xy: np.ndarray):
    """
    计算点集内部最近邻距离均值，用于粗略判断分辨率
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
    优先从 obs['umi'] 或 obs['total_counts'] 读取；
    如果没有，就退回到 X.sum(axis=1)
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
    根据空间坐标自动估计 scatter 的 spot_size (参数 s)。
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
    自动生成输出 h5ad 路径：
    - high_to_low: 用 high 分辨率 h5ad 的文件名，前加 reassigned_
    - low_to_high: 用 low 分辨率 h5ad 的文件名，前加 reassigned_
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
    把字符串标签聚合到 target spot 上。
    返回：
    1) mode_values: 每个 target spot 的众数标签，没有则 NA
    2) all_values: 每个 target spot 对应的所有标签拼接字符串
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
    每个 spot 画成方块，颜色表示每个 spot 的 UMI。
    """
    xy = get_spatial_from_adata(adata, spatial_key)
    umi, value_src = get_umi_values(adata)

    xy, umi = filter_valid_coords_values(xy, umi)

    if umi.size == 0:
        raise ValueError("没有可用于绘图的有效 spot。")

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
        print(f"自动估计 spot_size = {spot_size:.2f}")
    else:
        print(f"使用手动 spot_size = {spot_size:.2f}")

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

    print(f"✅ UMI 可视化已保存：{out_png}")


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
    使用两个 h5ad 的指定坐标键，自动判断高/低分辨率后建立映射
    """
    if reassignment_direction not in {"low_to_high", "high_to_low"}:
        raise ValueError("reassignment_direction 只能是 'low_to_high' 或 'high_to_low'。")

    xy1 = get_spatial_from_adata(adata_s1, s1_spatial_key)
    xy2 = get_spatial_from_adata(adata_s2, s2_spatial_key)

    def clean_xy(xy):
        mask = np.isfinite(xy).all(axis=1)
        return xy[mask, :], mask

    xy1_clean, mask1 = clean_xy(xy1)
    xy2_clean, mask2 = clean_xy(xy2)

    if xy1_clean.shape[0] == 0 or xy2_clean.shape[0] == 0:
        raise ValueError("S1 或 S2 有效坐标为空（全是 NA/Inf？）。")

    print(f"S1 有效坐标点数: {xy1_clean.shape[0]} / {xy1.shape[0]}")
    print(f"S2 有效坐标点数: {xy2_clean.shape[0]} / {xy2.shape[0]}")

    mean_s1, nn_s1 = mean_internal_nn_distance(xy1_clean)
    mean_s2, nn_s2 = mean_internal_nn_distance(xy2_clean)

    print(f"S1 内部最近邻距离均值: {mean_s1:.4f}")
    print(f"S2 内部最近邻距离均值: {mean_s2:.4f}")

    # 最近邻均值越大 -> 越稀疏 -> 越低分辨率
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

    print(f"\n自动判断：{low_res_name} = 低分辨率，{high_res_name} = 高分辨率")

    d_ref_max = float(np.max(nn_low)) if nn_low.size > 0 else 0.0
    print(f"低分辨率切片内部最近邻最大距离 d_ref_max = {d_ref_max:.4f}")

    low_indices_all = np.where(low_mask)[0]
    high_indices_all = np.where(high_mask)[0]

    tree_low = cKDTree(low_xy)
    dist, idx = tree_low.query(high_xy, k=1)

    if d_ref_max > 0:
        valid = dist <= 2.0 * d_ref_max
    else:
        valid = np.ones_like(dist, dtype=bool)

    n_drop = int(np.sum(~valid))
    print(f"距离过滤：删除 {n_drop} 个高分辨率点（dist > 2 * d_ref_max）。")

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
        print("\n当前模式：low_to_high（低分辨率表达 -> 高分辨率坐标）")
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
        print("\n当前模式：high_to_low（高分辨率表达加和 -> 低分辨率坐标）")

    print("\n映射表前几行：")
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
    根据 mapping + 两个 h5ad 构建新的 h5ad

    reserved_col 逻辑：
    - low_to_high: 新 h5ad 本来保留 high，所以 reserved_col 从 low 保留
    - high_to_low: 新 h5ad 本来保留 low，所以 reserved_col 从 high 保留
    """
    low_res_name = meta["low_res_name"]
    high_res_name = meta["high_res_name"]
    reassignment_direction = meta["reassignment_direction"]

    adata_low = adata_s1 if low_res_name == "S1" else adata_s2
    adata_high = adata_s2 if high_res_name == "S2" else adata_s1

    low_spatial_key = meta["s1_spatial_key"] if low_res_name == "S1" else meta["s2_spatial_key"]
    high_spatial_key = meta["s1_spatial_key"] if high_res_name == "S1" else meta["s2_spatial_key"]

    if reassignment_direction == "low_to_high":
        print("\n开始构建 low_to_high 的新 h5ad ...")

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

        # 新坐标是 high，所以 obs 主体来自 high
        obs_new = adata_high.obs.iloc[tgt_idx].copy()
        obs_new["reassigned_from"] = src_idx
        obs_new["reassigned_to"] = tgt_idx

        # reserved_col 在 low_to_high 时应该保留 low 的列
        if reserved_col is not None:
            if reserved_col not in adata_low.obs.columns:
                raise KeyError(
                    f"reserved_col='{reserved_col}' 不在 low 切片的 obs 中。"
                    f"可用列：{list(adata_low.obs.columns)}"
                )
            obs_new[f"reserved_low_{reserved_col}"] = (
                adata_low.obs.iloc[src_idx][reserved_col].astype(str).to_numpy()
            )

        var_new = adata_low.var.copy()
        spatial_new = get_spatial_from_adata(adata_high, high_spatial_key)[tgt_idx]

        adata_new = AnnData(X=X_new, obs=obs_new, var=var_new)
        adata_new.obsm["spatial"] = spatial_new

    else:
        print("\n开始构建 high_to_low 的新 h5ad ...")

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

        # 新坐标是 low，所以 obs 主体来自 low
        obs_new = adata_low.obs.copy()
        obs_new["mapped_high_count"] = mapped_counts.astype(int)

        # reserved_col 在 high_to_low 时应该保留 high 的列
        if reserved_col is not None:
            if reserved_col not in adata_high.obs.columns:
                raise KeyError(
                    f"reserved_col='{reserved_col}' 不在 high 切片的 obs 中。"
                    f"可用列：{list(adata_high.obs.columns)}"
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

    print("写出前 X 类型:", type(adata_new.X))

    out_dir = os.path.dirname(out_h5ad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    adata_new.write_h5ad(out_h5ad, compression="gzip")
    print(f"✅ 新 h5ad 已保存：{out_h5ad}")

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
    print("读取 h5ad ...")
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
    print(f"\n自动输出 h5ad 路径：{out_h5ad}")

    if map_csv is not None:
        out_dir_csv = os.path.dirname(map_csv)
        if out_dir_csv:
            os.makedirs(out_dir_csv, exist_ok=True)
        mapping.to_csv(map_csv, index=False)
        print(f"\n中间映射表已保存：{map_csv}")

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
            "SPOmiAlign reassignment（纯 h5ad 版本）：\n"
            "自动判断 S1/S2 哪个是高分辨率/低分辨率，并支持两种方向：\n"
            "1) low_to_high: 低分辨率表达分配到高分辨率坐标（1/k 分摊）\n"
            "2) high_to_low: 高分辨率表达聚合到低分辨率坐标（保留全部 low spots，未映射点补 0）\n"
            "支持为 S1/S2 分别指定坐标键。\n"
            "输出 h5ad 文件名自动生成，只需要传 out_dir。\n"
            "reserved_col 规则：low_to_high 从 low 保留；high_to_low 从 high 保留。"
        )
    )
    parser.add_argument("--s1_h5ad", "-h1", required=True, help="S1 h5ad 路径")
    parser.add_argument("--s2_h5ad", "-h2", required=True, help="S2 h5ad 路径")
    parser.add_argument("--out_dir", "-o", required=True, help="输出目录，程序自动命名新 h5ad")
    parser.add_argument("--map_csv", "-m", default=None, help="中间映射表 CSV 输出路径（可选）")

    parser.add_argument(
        "--s1_spatial_key",
        default="spatial",
        help="S1 使用的坐标键名，例如 spatial / spatial_raw / spatial_spomialign",
    )
    parser.add_argument(
        "--s2_spatial_key",
        default="spatial",
        help="S2 使用的坐标键名，例如 spatial / spatial_raw / spatial_spomialign",
    )

    parser.add_argument("--no_scale", action="store_true", help="关闭 1/k 缩放（仅对 low_to_high 生效）")

    parser.add_argument(
        "--reassignment_direction",
        choices=["low_to_high", "high_to_low"],
        default="low_to_high",
        help="重分配方向：low_to_high 或 high_to_low",
    )

    parser.add_argument(
        "--reserved_col",
        default=None,
        help="需要额外保留的另一侧切片 obs 列名，例如 Manual_annotation；不传则不保留",
    )

    parser.add_argument("--save_plots", action="store_true", help="保存输入/输出 h5ad 的 UMI 可视化图")
    parser.add_argument("--plot_dir", default=None, help="绘图输出目录（默认 out_dir 下的 reassignment_plots）")
    parser.add_argument(
        "--plot_spot_size",
        type=float,
        default=None,
        help="绘图时每个 spot 的大小（scatter 的 s）。不传则自动根据分辨率估计。",
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