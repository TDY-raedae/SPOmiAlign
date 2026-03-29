#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scanpy as sc
from PIL import Image
import cv2


# =========================
# Utility: intensity processing (optional log transform, 1-99 percentile clipping,
# normalization to [0,1], and optional threshold-based point filtering)
# =========================
def _prepare_intensity(
    intensity: np.ndarray,
    *,
    intensity_log_transform: bool = False,
    threshold_percentile: float | None = None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      intensity_norm: [0,1] float array
      keep_mask: boolean mask (all True when threshold_percentile=None)
    """
    v = np.asarray(intensity, dtype=np.float64)

    # Optional log1p transform. Use log1p consistently to avoid issues around 0.
    if intensity_log_transform:
        v = np.log1p(np.maximum(v, 0.0))

    # 1-99 percentile clipping
    p1 = float(np.nanpercentile(v, clip_low))
    p99 = float(np.nanpercentile(v, clip_high))
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 <= p1):
        vmin = float(np.nanmin(v)) if np.isfinite(np.nanmin(v)) else 0.0
        vmax = float(np.nanmax(v)) if np.isfinite(np.nanmax(v)) else (vmin + 1e-9)
        p1, p99 = vmin, vmax + 1e-9

    v_clip = np.clip(v, p1, p99)
    v_norm = (v_clip - p1) / (p99 - p1 + 1e-12)
    v_norm = np.clip(v_norm, 0.0, 1.0)

    # Optional thresholding based on the percentile of v_norm
    if threshold_percentile is None:
        keep = np.ones(v_norm.shape[0], dtype=bool)
    else:
        thr = float(np.nanpercentile(v_norm, float(threshold_percentile)))
        keep = v_norm > thr

    return v_norm.astype(np.float32), keep


# =========================
# Utility: point kernel template (circle / square)
# =========================
def _make_kernel(radius: int, shape: str = "circle") -> np.ndarray:
    r = int(max(0, radius))
    if r == 0:
        return np.ones((1, 1), dtype=np.float32)

    size = 2 * r + 1
    if shape.lower() in ("square", "rect", "box"):
        return np.ones((size, size), dtype=np.float32)

    # circle
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = (x * x + y * y) <= (r * r)
    return mask.astype(np.float32)


# =========================
# Utility: grayscale enhancement (CLAHE + gamma + unsharp masking)
# for the final uint8 grayscale image
# =========================
def enhance_gray_uint8(
    gray_uint8: np.ndarray,
    clahe_clip: float = 4.0,
    clahe_grid: tuple[int, int] = (8, 8),
    gamma: float = 0.8,
    unsharp_ksize: tuple[int, int] = (5, 5),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,
) -> np.ndarray:
    """
    Input/output: uint8 grayscale image (H, W)
    gamma < 1 brightens the image; gamma > 1 darkens it
    """
    if gray_uint8.dtype != np.uint8:
        raise ValueError("enhance_gray_uint8 expects uint8 image")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    g = clahe.apply(gray_uint8)

    # Gamma correction (keep the same convention as before: invGamma = 1/gamma)
    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    g = cv2.LUT(g, table)

    # Unsharp
    blurred = cv2.GaussianBlur(g, unsharp_ksize, unsharp_sigma)
    g = cv2.addWeighted(g, float(unsharp_amount), blurred, -float(unsharp_amount - 1.0), 0)

    return g


# =========================
# Utility: clockwise rotation + uniform scaling in the mathematical
# coordinate system, optional
# =========================
def _apply_rotate_scale_clockwise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    origin_mode: str = "data",  # 'data' (centroid) | 'center' (bounding-box center) | 'zero'
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([x, y]).T.astype(np.float64)

    if origin_mode == "data":
        origin = pts.mean(axis=0)
    elif origin_mode == "center":
        origin = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2.0,
                           (pts[:, 1].min() + pts[:, 1].max()) / 2.0], dtype=np.float64)
    elif origin_mode == "zero":
        origin = np.array([0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError("origin_mode must be 'data'|'center'|'zero'")
    # Clockwise rotation: theta -> -theta
    th = np.deg2rad(-float(rotate_deg))
    c, s = np.cos(th), np.sin(th)
    R_cw = np.array([[c,  s],
                     [-s, c]], dtype=np.float64)

    A = float(scale) * R_cw
    out = (pts - origin.reshape(1, 2)) @ A.T + origin.reshape(1, 2)
    return out[:, 0], out[:, 1], origin


# =========================
# Utility: automatically scale to (1152, 864) and shift negative values to >= 0
# (always applied, no extra parameters needed)
# =========================
def _auto_scale_to_canvas_and_shift_nonnegative(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_w: float = 1152.0,
    target_h: float = 864.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Automatic coordinate processing (no user-facing parameters):

    A) Uniform upscaling rule (the same scale is applied to both x and y):
       1) If xmax > target_w and ymax > target_h: do not scale
       2) If only xmax <= target_w: scale = target_w / xmax
       3) If only ymax <= target_h: scale = target_h / ymax
       4) If xmax <= target_w and ymax <= target_h:
          scale = max(target_w/xmax, target_h/ymax)

    B) If negative values exist: shift everything so coordinates become >= 0
       x += -min(x) if min(x)<0
       y += -min(y) if min(y)<0

    Note: this step does not force the output canvas size to be 1152x864.
    The PNG size is still determined by the maximum transformed coordinates.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    if x.size == 0:
        return x, y

    eps = 1e-12
    xmax = float(np.nanmax(x))
    ymax = float(np.nanmax(y))

    need_scale_x = (xmax <= target_w)
    need_scale_y = (ymax <= target_h)

    if need_scale_x or need_scale_y:
        sx = target_w / max(xmax, eps)
        sy = target_h / max(ymax, eps)
        if need_scale_x and need_scale_y:
            s = max(sx, sy)
        elif need_scale_x:
            s = sx
        else:
            s = sy
        x *= s
        y *= s

    xmin = float(np.nanmin(x))
    ymin = float(np.nanmin(y))
    if xmin < 0:
        x += (-xmin)
    if ymin < 0:
        y += (-ymin)

    return x, y
def _only_shift_nonnegative(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift negative values to >= 0 only (no user-facing parameters):
       x += -min(x) if min(x)<0
       y += -min(y) if min(y)<0
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    if x.size == 0:
        return x, y

    xmin = float(np.nanmin(x))
    ymin = float(np.nanmin(y))
    if xmin < 0:
        x += (-xmin)
        print("x add offset")
    if ymin < 0:
        y += (-ymin)
        print("y add offset")

    return x, y


# =========================
# Core: h5ad -> pixel-level rasterized grayscale image
# with optional enhancement
# =========================
def rasterize_h5ad_to_image(
    *,
    input_h5ad: str,
    output_png: str,

    # Coordinate source: by default, use the first two columns of obsm['spatial']
    spatial_key: str = "spatial",
    x_obs_col: str | None = None,
    y_obs_col: str | None = None,

    # Intensity: use X.sum(axis=1) by default. To use an obs column instead,
    # set intensity_mode="obs_col" and provide intensity_obs_col.
    intensity_mode: str = "X_sum",       # "X_sum" | "obs_col"
    intensity_obs_col: str | None = None,

    # Unified intensity handling: optional log1p, shared 1-99 percentile clipping,
    # normalization, and optional threshold-based point filtering
    intensity_log_transform: bool = False,
    threshold_percentile: float | None = None,  # For example 80; None disables thresholding

    # Overlay rule (default: white background with black points)
    background: str = "white",  # "white" | "black"

    # Point shape and size
    point_shape: str = "circle",  # "circle" | "square"
    radius: int = 5,

    # Optional enhancement before output (CLAHE + gamma + unsharp masking)
    enhance: bool = False,
    clahe_clip: float = 4.0,
    clahe_grid: tuple[int, int] = (8, 8),
    gamma: float = 0.8,
    unsharp_ksize: tuple[int, int] = (5, 5),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,

    # Optional user-specified rotate/scale settings
    rotate: float = 0.0,            # Clockwise rotation angle
    scale: float = 1.0,             # Uniform scaling factor
    rotate_origin: str = "data",    # 'data'|'center'|'zero'

    canvas_size: tuple[int, int] | None = None,  # Added: (width, height)
):
    """
    Output:
      - output_png: pixel-level rasterized grayscale image
        (default: white background with black points)
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    # ===== Read coordinates =====
    if x_obs_col is not None and y_obs_col is not None:
        x = np.asarray(adata.obs[x_obs_col], dtype=np.float64)
        y = np.asarray(adata.obs[y_obs_col], dtype=np.float64)
    else:
        if spatial_key not in adata.obsm:
            raise KeyError(
                f"obsm['{spatial_key}'] does not exist; please provide x_obs_col/y_obs_col "
                "to specify coordinate columns"
            )
        xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"obsm['{spatial_key}'] must have shape (N,2+); got {xy.shape}")
        x, y = xy[:, 0], xy[:, 1]

    # ===== Read raw intensity =====
    if intensity_mode == "X_sum":
        # Compatible with sparse matrices
        try:
            intensity_raw = np.array(adata.X.sum(axis=1)).reshape(-1)
        except Exception:
            intensity_raw = np.asarray(adata.X.sum(axis=1)).ravel()
    elif intensity_mode == "obs_col":
        if not intensity_obs_col:
            raise ValueError("intensity_obs_col must be provided when intensity_mode='obs_col'")
        intensity_raw = np.asarray(adata.obs[intensity_obs_col], dtype=np.float64)
    else:
        raise ValueError("intensity_mode must be 'X_sum' or 'obs_col'")

    # ===== Unified intensity processing: optional log, 1-99 clipping,
    # normalization, and optional thresholding =====
    intensity_norm, keep_mask = _prepare_intensity(
        intensity_raw,
        intensity_log_transform=bool(intensity_log_transform),
        threshold_percentile=threshold_percentile,
        clip_low=1.0,
        clip_high=99.0,
    )

    # Optional: keep these values in adata.obs for debugging or downstream use,
    # even though they are not written back to h5ad here
    adata.obs["render_intensity_raw"] = intensity_raw
    adata.obs["render_intensity_norm"] = intensity_norm
    adata.obs["render_keep"] = keep_mask.astype(np.int8)

    # ===== Filter valid points (valid coordinates/intensity and keep=True) =====
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(intensity_norm) & keep_mask
    x = x[valid]
    y = y[valid]
    v = intensity_norm[valid]

    if x.size == 0:
        raise ValueError(
            "The number of valid points is 0. Please check the coordinate/intensity columns, "
            "or lower threshold_percentile."
        )
    origin=None
    # ===== Coordinates: optional user-specified clockwise rotation + scaling =====
    if (abs(float(rotate)) > 1e-12) or (abs(float(scale) - 1.0) > 1e-12):
        x, y,origin = _apply_rotate_scale_clockwise(
            x, y,
            rotate_deg=float(rotate),
            scale=float(scale),
            origin_mode=str(rotate_origin),
        )

    # ===== Coordinates: automatically scale to 1152x864 and shift negatives to >= 0
    # (always applied, no parameters needed) =====
    # x, y = _auto_scale_to_canvas_and_shift_nonnegative(x, y)
    

    # ===== Convert coordinates to pixels (round to the nearest pixel) =====
    x_pix = np.rint(x).astype(np.int32)
    y_pix = np.rint(y).astype(np.int32)

    if canvas_size is not None:
        # User-specified resolution (W, H)
        W, H = canvas_size
    else:
        # Automatically compute resolution
        W = int(x_pix.max()) + 1
        H = int(y_pix.max()) + 1

    if W <= 0 or H <= 0:
        raise ValueError(f"Invalid canvas size: W={W}, H={H}")

    # ===== Background and the darker/brighter point rule =====
    # Always use v in [0,1]:
    # - white background with black points: background=1.0, larger values mean darker points
    # - black background with white points: background=0.0, larger values mean brighter points
    bg = background.lower()
    if bg not in ("white", "black"):
        raise ValueError("background must be 'white' or 'black'")

    if bg == "white":
        img = np.ones((H, W), dtype=np.float32)
        vals = 1.0 - v
        take_darker = True   # Use min for overlay so darker values win
    else:
        img = np.zeros((H, W), dtype=np.float32)
        vals = v
        take_darker = False  # Use max for overlay so brighter values win

    # ===== Point kernel template (circle / square) =====
    kernel = _make_kernel(radius=int(radius), shape=str(point_shape))
    r = int(max(0, radius))

    # ===== Rasterized overlay (pixel-level stamping) =====
    for xv, yv, val in zip(x_pix, y_pix, vals):
        # patch bbox
        r0 = yv - r
        c0 = xv - r
        r1 = yv + r + 1
        c1 = xv + r + 1

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(H, r1)
        cc1 = min(W, c1)

        if rr0 >= rr1 or cc0 >= cc1:
            continue

        kr0 = rr0 - r0
        kc0 = cc0 - c0
        kr1 = kr0 + (rr1 - rr0)
        kc1 = kc0 + (cc1 - cc0)

        patch = float(val) * kernel[kr0:kr1, kc0:kc1]

        if take_darker:
            # White background with black points: use min so darker values overwrite
            img[rr0:rr1, cc0:cc1] = np.minimum(img[rr0:rr1, cc0:cc1], patch)
        else:
            # Black background with white points: use max so brighter values overwrite
            img[rr0:rr1, cc0:cc1] = np.maximum(img[rr0:rr1, cc0:cc1], patch)

    # ===== Save PNG =====
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    if enhance:
        img_u8 = enhance_gray_uint8(
            img_u8,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            gamma=gamma,
            unsharp_ksize=unsharp_ksize,
            unsharp_sigma=unsharp_sigma,
            unsharp_amount=unsharp_amount,
        )

    Image.fromarray(img_u8, mode="L").save(output_png)
    print(
        f"[OK] PNG saved: {output_png} "
        f"({W}x{H}, background={background}, shape={point_shape}, radius={radius})"
    )

    return output_png,origin

# =========================
# Usage examples
# =========================
if __name__ == "__main__":
    # Example 1: MERFISH (default spatial_key='spatial', default intensity_mode='X_sum')
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Zhuang-ABCA-3-log2-metadata_08.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/merfish_008.png",
        background="black",
        point_shape="circle",
        radius=1,
        threshold_percentile=None,      # No thresholding
        intensity_log_transform=False,  # X is already log2-transformed, so no extra log is needed
        enhance=True,
        rotate=0.0,
        scale=1.0,
    )

    # Example 2: Slide-seq 43 (use an obs column as intensity and keep points above the 80th percentile)
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_43.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_43.png",
        x_obs_col="Raw_Slideseq_X",
        y_obs_col="Raw_Slideseq_Y",
        # intensity_mode="obs_col",
        intensity_obs_col="nFeature_Spatial",
        intensity_log_transform=True,   # log1p is recommended for nFeature counts
        threshold_percentile=80,        # Keep points above the 80th percentile
        background="black",
        point_shape="circle",
        radius=5,
        enhance=True,                  # Reuse the existing enhancement pipeline
        rotate=90,
        scale=1.0,
    )

    # Example 3: Slide-seq 29 (use an obs column as intensity and keep points above the 80th percentile)
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/data_preprocessing/Puck_Num_29.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/Puck_Num_29.png",
        x_obs_col="Raw_Slideseq_X",
        y_obs_col="Raw_Slideseq_Y",
        # intensity_mode="obs_col",
        intensity_obs_col="nFeature_Spatial",
        intensity_log_transform=True,   # log1p is recommended for nFeature counts
        threshold_percentile=80,        # Keep points above the 80th percentile
        background="black",
        point_shape="circle",
        radius=5,
        enhance=True,                  # Reuse the existing enhancement pipeline
        rotate=180,
        scale=1.0,
    )
    # Example 4: SM, ST, and SP
    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-PLATO.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/sp.png",
        background="white",
        point_shape="square",
        radius=15,
        threshold_percentile=None,      # No thresholding
        intensity_log_transform=False,  # X is already log2-transformed, so no extra log is needed
        enhance=False,
        rotate=0.0,
        scale=1.0,
    )

    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-MAGIC-seq.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/st.png",
        background="white",
        point_shape="square",
        radius=15,
        threshold_percentile=None,      # No thresholding
        intensity_log_transform=False,  # X is already log2-transformed, so no extra log is needed
        enhance=False,
        rotate=0.0,
        scale=1.0,
    )

    rasterize_h5ad_to_image(
        input_h5ad="/mnt/A3/ivy/register_data/3omics/Cerebellum-MALDI-MSI.h5ad",
        output_png="/mnt/A3/ivy/register_data/SPOmiAlign_Repro/output_image/sm.png",
        background="white",
        point_shape="square",
        radius=12,
        threshold_percentile=None,      # No thresholding
        intensity_log_transform=False,  # X is already log2-transformed, so no extra log is needed
        enhance=False,
        rotate=60.0,
        scale=0.6,
    )
