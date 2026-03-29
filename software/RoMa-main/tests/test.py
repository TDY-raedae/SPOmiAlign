import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Core update: use the same model interface as demo_match_aligen.py ---
from romatch import roma_outdoor

def evaluate_snr(certainty_map, img_path):
    """
    Compute SNR: mean certainty in edge regions / mean certainty in flat regions
    certainty_map: (H, W) numpy array, range [0, 1]
    """
    # Read the original image to generate reference edges (ground truth)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        print(f"Failed to read image: {img_path}")
        return 0, 0, 0
    
    # Resize to match the certainty map (the model may resize the input internally)
    h, w = certainty_map.shape
    img_resized = cv2.resize(img, (w, h))
    
    # Use the Canny operator to extract physical edges
    edges = cv2.Canny(img_resized, 100, 200)
    mask_edge = edges > 127
    mask_flat = ~mask_edge
    
    # Compute regional means
    mean_edge = certainty_map[mask_edge].mean() if mask_edge.any() else 0
    mean_flat = certainty_map[mask_flat].mean() if mask_flat.any() else 0
    
    # SNR: edge response relative to flat-region response
    snr = mean_edge / (mean_flat + 1e-6)
    return mean_edge, mean_flat, snr

def main():
    # 1. Configure parameters
    # Replace these with your actual test image paths
    imA_path = "/data/Newdisk/Bigmodel/zxm/Match/RoMa-main/RoMa-main/Dataset/simulated_MISAR/E15_5-S1-HE.jpg"
    imB_path = "/data/Newdisk/Bigmodel/zxm/Match/RoMa-main/RoMa-main/Dataset/simulated_MISAR/new/E15_5-S2-HE_warped_rt15.png"
    
    out_dir = "outputs_evaluation_roma_outdoor"
    os.makedirs(out_dir, exist_ok=True)

    # Fourier edge-weighting parameters
    edge_params = {
        "edge_sigma": 25,    # Controls the frequency range used for edge extraction
        "edge_decay": 15.0,  # Controls how quickly the weights decay with distance
        "edge_power": 2.0    # Weight exponent; values > 0 enable weighting
    }

    # 2. Device setup and model loading (see demo_match_aligen.py)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading RoMa outdoor model...")
    # Use the high-resolution setting to stay consistent with the demo
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    
    # Get the expected model output resolution
    H_out, W_out = roma_model.get_output_resolution()
    print(f"Model output resolution: {H_out}x{W_out}")

    # ---------------------------------------------------------
    # 3. Baseline inference: disable edge weighting
    # ---------------------------------------------------------
    print("\nRunning Baseline Inference (edge_power=0.0)...")
    with torch.no_grad():
        # Pass file paths directly; the model handles loading and resizing internally
        # Pass edge_power=0.0 to disable the weighting logic
        warp_base, cert_base = roma_model.match(
            imA_path, imB_path, 
            device=device, 
            edge_power=0.0 
        )
        
        # --- Added: sample matched points (baseline) ---
        # Sample 5000 points
        matches_base, _ = roma_model.sample(warp_base, cert_base, num=5000)
        # Get pixel coordinates (note: this requires the internally resized dimensions, usually H_out and W_out)
        # roma_model.match internally resizes the image to (H_out, W_out)
        kptsA_base, kptsB_base = roma_model.to_pixel_coordinates(matches_base, H_out, W_out, H_out, W_out)
        kptsA_base = kptsA_base.cpu().numpy()
        kptsB_base = kptsB_base.cpu().numpy()
        # -----------------------------------

        # Handle the output shape
        # roma_outdoor uses symmetric=True by default, so certainty is usually shaped like (B, 1, H, 2*W) or (H, 2*W)
        cert_base = cert_base.squeeze() # -> (H, 2*W)
        
        # Extract the left half (the certainty map corresponding to imA)
        if cert_base.shape[-1] == 2 * W_out:
            cert_base_np = cert_base[:, :W_out].cpu().numpy()
        else:
            # Fallback in case the model configuration changes and produces non-symmetric output
            cert_base_np = cert_base.cpu().numpy()

    # ---------------------------------------------------------
    # 4. Weighted inference: enable edge weighting
    # ---------------------------------------------------------
    print(f"Running Weighted Inference (edge_power={edge_params['edge_power']})...")
    with torch.no_grad():
        # Pass the additional parameters introduced in matcher.py
        warp_weighted, cert_weighted = roma_model.match(
            imA_path, imB_path, 
            device=device, 
            edge_power=edge_params['edge_power'],
            edge_sigma=edge_params['edge_sigma'],
            edge_decay=edge_params['edge_decay']
        )
        
        # --- Added: sample matched points (weighted) ---
        matches_weighted, _ = roma_model.sample(warp_weighted, cert_weighted, num=5000)
        kptsA_weighted, kptsB_weighted = roma_model.to_pixel_coordinates(matches_weighted, H_out, W_out, H_out, W_out)
        kptsA_weighted = kptsA_weighted.cpu().numpy()
        kptsB_weighted = kptsB_weighted.cpu().numpy()
        # -----------------------------------
        
        cert_weighted = cert_weighted.squeeze()
        
        # Extract the left half in the same way
        if cert_weighted.shape[-1] == 2 * W_out:
            cert_weighted_np = cert_weighted[:, :W_out].cpu().numpy()
        else:
            cert_weighted_np = cert_weighted.cpu().numpy()

    # ---------------------------------------------------------
    # 5. Compute evaluation metrics (SNR)
    # ---------------------------------------------------------
    print("\n--- Evaluation Results ---")
    
    # Evaluate the baseline result
    m_edge_b, m_flat_b, snr_b = evaluate_snr(cert_base_np, imA_path)
    print(f"[Baseline] Edge Mean: {m_edge_b:.4f}, Flat Mean: {m_flat_b:.4f}")
    print(f"[Baseline] SNR (Edge/Flat): {snr_b:.4f}")
    
    # Evaluate the weighted result
    m_edge_w, m_flat_w, snr_w = evaluate_snr(cert_weighted_np, imA_path)
    print(f"[Weighted] Edge Mean: {m_edge_w:.4f}, Flat Mean: {m_flat_w:.4f}")
    print(f"[Weighted] SNR (Edge/Flat): {snr_w:.4f}")
    
    # Compute the percentage improvement
    if snr_b > 0:
        improvement = (snr_w - snr_b) / snr_b * 100
        print(f"SNR Improvement: {improvement:.2f}%")
    else:
        print("SNR Improvement: N/A (Baseline SNR is 0)")

    # ---------------------------------------------------------
    # 6. Save visualizations
    # ---------------------------------------------------------
    print("Generating visualizations...")
    
    # Read the original image for overlay (convert to grayscale first, then back to RGB, to keep the background subtle)
    img_A = cv2.imread(imA_path)
    img_B = cv2.imread(imB_path) # Read image B for drawing match lines
    
    if img_A is None:
        img_A = np.zeros((cert_base_np.shape[0], cert_base_np.shape[1], 3), dtype=np.uint8)
    if img_B is None:
        img_B = np.zeros((cert_base_np.shape[0], cert_base_np.shape[1], 3), dtype=np.uint8)
    
    # Resize the original image to match the output size (H_out, W_out)
    # Note: cert_base_np already has shape (H_out, W_out)
    h, w = cert_base_np.shape
    img_A_resized = cv2.resize(img_A, (w, h))
    img_B_resized = cv2.resize(img_B, (w, h)) # Resize image B as well
    
    # Convert to a grayscale background to avoid distracting colors behind the heatmap
    img_A_gray = cv2.cvtColor(img_A_resized, cv2.COLOR_BGR2GRAY)
    img_A_bgr = cv2.cvtColor(img_A_gray, cv2.COLOR_GRAY2BGR)

    def create_overlay(certainty_map, bg_img):
        # 1. Dynamic min-max normalization (increase contrast)
        c_min = certainty_map.min()
        c_max = certainty_map.max()
        
        # Avoid division by zero
        if c_max - c_min < 1e-6:
            norm = np.zeros_like(certainty_map)
        else:
            norm = (certainty_map - c_min) / (c_max - c_min)
            
        # 2. Clip again for safety
        norm = np.clip(norm, 0, 1)
        
        # 3. Convert to uint8
        norm_uint8 = (norm * 255).astype(np.uint8)
        
        # 4. Generate a heatmap (JET: blue=low, red=high)
        heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        
        # 5. Overlay: 0.6 * heatmap + 0.4 * original image
        overlay = cv2.addWeighted(heatmap, 0.6, bg_img, 0.4, 0)
        
        # Print summary statistics for debugging
        print(f"  [Map Stats] Min: {c_min:.4f}, Max: {c_max:.4f} -> Normalized to [0, 1]")
        
        return overlay, heatmap

    # --- Added: helper to draw match lines ---
    def draw_matches_side_by_side(img1, img2, kpts1, kpts2, name="matches"):
        """
        img1, img2: (H, W, 3) BGR images
        kpts1, kpts2: (N, 2) numpy arrays, coordinates in (x, y)
        """
        h, w = img1.shape[:2]
        # Create a side-by-side canvas
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = img1
        vis[:, w:] = img2
        
        # Random colors
        # Draw connection lines
        # To avoid clutter, draw only the first 200 points or a random subset
        num_draw = min(len(kpts1), 200)
        indices = np.random.choice(len(kpts1), num_draw, replace=False)
        
        for idx in indices:
            pt1 = (int(kpts1[idx, 0]), int(kpts1[idx, 1]))
            pt2 = (int(kpts2[idx, 0] + w), int(kpts2[idx, 1])) # Shift the x coordinate by w for the image on the right
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(vis, pt1, 2, color, -1)
            cv2.circle(vis, pt2, 2, color, -1)
            
        return vis

    # --- Added: draw dense matched-point distributions (without lines) ---
    def draw_dense_matches(img, kpts, color=(0, 255, 0), radius=1, alpha=0.4):
        """
        Draw all matched points on a single image and use alpha blending to show density
        img: (H, W, 3) BGR
        kpts: (N, 2)
        """
        # Create an overlay layer
        overlay = img.copy()
        print(len(kpts))
        # Draw points in a loop (OpenCV circle does not support batch drawing, but this is still fast for a few thousand points)
        # For speed and visual quality, we could manipulate pixels directly or use matplotlib; here we simulate it with cv2
        
        for pt in kpts:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), radius, color, -1)
            
        # Blend the original image with the overlay to create transparency
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # --- Added: visualize correspondences with color coding ---
    def draw_color_coded_matches(img1, img2, kpts1, kpts2, radius=2):
        """
        Use position-based colors to visualize correspondences.
        1. Generate colors from the positions of kpts1 (HSV space: x->hue, y->saturation/value).
        2. Draw kpts2 with the same colors.
        This makes it easy to see which region in image 1 each matched point came from.
        """
        h, w = img1.shape[:2]
        
        # Create the canvas
        vis1 = img1.copy()
        vis2 = img2.copy()
        
        # Normalize coordinates to generate colors
        # x_norm: 0~1, y_norm: 0~1
        x_norm = kpts1[:, 0] / w
        y_norm = kpts1[:, 1] / h
        
        # Generate colors (HSV -> BGR)
        # Hue: determined by the x coordinate (0~179)
        # Saturation: fixed at 255 for vivid colors
        # Value: determined by the y coordinate (100~255 to avoid becoming too dark)
        
        # For speed, use NumPy batch operations and then convert to uint8
        hue = (x_norm * 179).astype(np.uint8)
        sat = np.ones_like(hue) * 255
        val = ((1 - y_norm) * 155 + 100).astype(np.uint8) # Larger y can be made darker, or the mapping can be inverted
        
        # Combine into an HSV image with shape (N, 1, 3)
        hsv_pixels = np.stack([hue, sat, val], axis=1).reshape(-1, 1, 3)
        # Convert to BGR
        bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        
        # Draw points
        for i, (pt1, pt2) in enumerate(zip(kpts1, kpts2)):
            color = bgr_pixels[i].tolist()
            # Filled circle
            cv2.circle(vis1, (int(pt1[0]), int(pt1[1])), radius, color, -1)
            cv2.circle(vis2, (int(pt2[0]), int(pt2[1])), radius, color, -1)
            
        # Concatenate
        return np.hstack((vis1, vis2))

    # --- Added: draw Top-K high-confidence matches ---
    def draw_topk_matches(img1, img2, kpts1, kpts2, scores, k=5000, radius=1):
        """
        Select the top-k points by score for drawing (with color coding)
        """
        if len(scores) > k:
            # Get the indices of the top-k highest scores
            # argsort returns ascending order, so take the last k values and reverse them
            top_indices = np.argsort(scores)[-k:][::-1]
            kpts1_top = kpts1[top_indices]
            kpts2_top = kpts2[top_indices]
        else:
            kpts1_top = kpts1
            kpts2_top = kpts2
            
        # Reuse the earlier color-coded drawing helper
        return draw_color_coded_matches(img1, img2, kpts1_top, kpts2_top, radius=radius)

    # 1. Get scores for all sampled baseline points (this would normally require resampling or changing the sample interface,
    # but roma_model.sample already returns sampled matches,
    # so we need to look up the corresponding certainty values)
    
    # To get an accurate Top-K set, read values directly from the original certainty map
    # Note: here we simply reuse the previously sampled 5000 points,
    # because roma_model.sample already performs weighted sampling; here we assume we want to inspect which of these 5000 points
    # are relatively higher-scoring, or simply visualize all 5000 points since they already represent high-confidence samples.
    
    # If you want to strictly select the Top 5000 points from the full image instead of sampling by probability,
    # you need to work with the certainty map directly. The following code shows how to do strict Top-K selection:

    def get_strict_topk_kpts(warp, certainty, H, W, k=5000):
        """
        Strictly select the k highest-certainty points from the full image
        """
        # Flatten the certainty map
        cert_flat = certainty.flatten() # (H*W,)
        
        # Get Top-K indices
        if cert_flat.numel() > k:
            topk_vals, topk_indices = torch.topk(cert_flat, k)
        else:
            topk_vals = cert_flat
            topk_indices = torch.arange(cert_flat.numel(), device=cert_flat.device)
            
        # Convert indices back to (y, x) coordinates
        # Note: certainty can have shape (B, 1, H, W) or (H, W)
        # Assume the input is already shaped like (H, W)
        h_map, w_map = certainty.shape[-2:]
        
        # Compute sampled-point coordinates on the feature map
        y_norm = (topk_indices // w_map).float() / (h_map - 1)
        x_norm = (topk_indices % w_map).float() / (w_map - 1)
        
        # Corresponding warp values (target-image coordinates)
        # warp shape: (B, 2, H, W) -> (2, H, W)
        warp_flat = warp.reshape(2, -1) # (2, H*W)
        warp_topk = warp_flat[:, topk_indices] # (2, k)
        
        # Source-image coordinates (normalize from -1~1 to 0~1)
        # Note: RoMa warp values are in -1~1, so they must be converted
        # Here we directly use grid coordinates as source-image coordinates
        kpts1_norm = torch.stack([x_norm, y_norm], dim=1) # (k, 2) range [0, 1]
        
        # Target-image coordinates (warp values are in -1~1)
        kpts2_norm = (warp_topk.permute(1, 0) + 1) / 2 # (k, 2) range [0, 1]
        
        # Convert to pixel coordinates
        kpts1_pixel = kpts1_norm * torch.tensor([W, H], device=kpts1_norm.device)
        kpts2_pixel = kpts2_norm * torch.tensor([W, H], device=kpts2_norm.device)
        
        return kpts1_pixel.cpu().numpy(), kpts2_pixel.cpu().numpy()

    print("Extracting Strict Top-5000 Matches by Certainty...")
    
    # Baseline Top-5000
    # Note: cert_base was previously squeezed into shape (H, 2*W), so it needs to be split back to (H, W)
    # warp_base must be handled consistently as well
    # It is safer to fetch the raw outputs again, or reuse the earlier logic
    # Assume cert_base_np is a NumPy array with shape (H, W); we need a torch tensor
    
    # For simplicity, directly reuse the previously computed cert_base (tensor) and warp_base
    # warp_base: (B, 2, H, W)
    # cert_base: (B, 1, H, W) or (H, W)
    
    # Ensure the dimensions are correct (take batch 0)
    if len(warp_base.shape) == 4:
        w_b = warp_base[0]
        c_b = cert_base if len(cert_base.shape)==3 else cert_base.unsqueeze(0) # Ensure a channel dimension exists
        if c_b.shape[0] != 1: c_b = c_b.unsqueeze(0) # (1, H, W)
    else:
        w_b = warp_base
        c_b = cert_base
        
    # Extract the left half because RoMa concatenates the output
    h_map, w_map = w_b.shape[-2:]
    w_half = w_map // 2
    
    # Extract the warp and certainty corresponding to the left image
    w_b_left = w_b[:, :, :w_half]
    c_b_left = c_b[:, :w_half] if c_b.dim() == 2 else c_b[:, :, :w_half]
    
    kptsA_top_base, kptsB_top_base = get_strict_topk_kpts(w_b_left, c_b_left, H_out, W_out, k=5000)
    
    # Weighted Top-5000
    if len(warp_weighted.shape) == 4:
        w_w = warp_weighted[0]
        c_w = cert_weighted if len(cert_weighted.shape)==3 else cert_weighted.unsqueeze(0)
    else:
        w_w = warp_weighted
        
    w_w_left = w_w[:, :, :w_half]
    c_w_left = c_w[:, :w_half] if c_w.dim() == 2 else c_w[:, :, :w_half]
    
    kptsA_top_weighted, kptsB_top_weighted = get_strict_topk_kpts(w_w_left, c_w_left, H_out, W_out, k=5000)

    # --- Update: Top-K selection based on already sampled points ---
    def get_topk_from_sampled(kpts1, kpts2, certainty_map, k=5000):
        """
        Select Top-K points from existing sampled points based on values from certainty_map
        kpts1: (N, 2) pixel coordinates
        certainty_map: (H, W) numpy array
        """
        h, w = certainty_map.shape
        scores = []
        valid_indices = []
        
        for i, pt in enumerate(kpts1):
            x, y = int(pt[0]), int(pt[1])
            # Boundary check
            if 0 <= x < w and 0 <= y < h:
                score = certainty_map[y, x]
                scores.append(score)
                valid_indices.append(i)
                
        scores = np.array(scores)
        valid_indices = np.array(valid_indices)
        
        # Select Top-K
        if len(scores) > k:
            # argsort returns ascending order, so take the last k values and reverse them for descending order
            top_k_idx_in_scores = np.argsort(scores)[-k:][::-1]
            final_indices = valid_indices[top_k_idx_in_scores]
        else:
            final_indices = valid_indices
            
        return kpts1[final_indices], kpts2[final_indices]

    print("Extracting Top-5000 Matches from Sampled Points...")
    
    # Baseline Top-5000 (selected from previously sampled points)
    # Note: the previous sample already contains 5000 points; to focus on a smaller high-confidence subset, set k=2000
    # Or, if more points were sampled earlier (for example 10000), select the Top 5000 here
    # This demonstrates score-based sorting from the existing kptsA_base set (sample is already weighted, but not strict Top-K)
    
    kptsA_top_base, kptsB_top_base = get_topk_from_sampled(kptsA_base, kptsB_base, cert_base_np, k=5000)
    
    # Weighted Top-5000
    kptsA_top_weighted, kptsB_top_weighted = get_topk_from_sampled(kptsA_weighted, kptsB_weighted, cert_weighted_np, k=5000)

    # Draw the Top-K color-coded visualization
    print("Drawing Top-5000 Sorted Matches...")
    vis_topk_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_base, kptsB_top_base)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_baseline.png"), vis_topk_base)
    
    vis_topk_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_weighted, kptsB_top_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_weighted.png"), vis_topk_weighted)
    
    print(f"  - matches_top5000_*.png: high-confidence matches selected from sampled points")

    # Draw the Top-K color-coded visualization
    print("Drawing Top-5000 Strict Certainty Matches...")
    vis_topk_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_base, kptsB_top_base)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_baseline.png"), vis_topk_base)
    
    vis_topk_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_top_weighted, kptsB_top_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_top5000_weighted.png"), vis_topk_weighted)
    
    print(f"  - matches_top5000_*.png: strictly selected top-5000 certainty points")

    # Draw baseline match lines
    print("Drawing Baseline Matches...")
    vis_matches_base = draw_matches_side_by_side(img_A_resized, img_B_resized, kptsA_base, kptsB_base)
    cv2.imwrite(os.path.join(out_dir, "matches_baseline.png"), vis_matches_base)
    
    # Draw weighted match lines
    print("Drawing Weighted Matches...")
    vis_matches_weighted = draw_matches_side_by_side(img_A_resized, img_B_resized, kptsA_weighted, kptsB_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_weighted.png"), vis_matches_weighted)
    # -----------------------------

    # Draw the baseline dense point distribution
    print("Drawing Baseline Dense Points...")
    # Points on image A (green)
    vis_kptsA_base = draw_dense_matches(img_A_resized, kptsA_base, color=(0, 255, 0))
    # Points on image B (red)
    vis_kptsB_base = draw_dense_matches(img_B_resized, kptsB_base, color=(0, 0, 255))
    
    # Concatenate for display
    vis_dense_base = np.hstack((vis_kptsA_base, vis_kptsB_base))
    cv2.imwrite(os.path.join(out_dir, "dense_points_baseline.png"), vis_dense_base)
    
    # Draw the weighted dense point distribution
    print("Drawing Weighted Dense Points...")
    vis_kptsA_weighted = draw_dense_matches(img_A_resized, kptsA_weighted, color=(0, 255, 0))
    vis_kptsB_weighted = draw_dense_matches(img_B_resized, kptsB_weighted, color=(0, 0, 255))
    
    vis_dense_weighted = np.hstack((vis_kptsA_weighted, vis_kptsB_weighted))
    cv2.imwrite(os.path.join(out_dir, "dense_points_weighted.png"), vis_dense_weighted)
    # -----------------------------

    # Draw the baseline color-coded view
    print("Drawing Baseline Color-Coded Matches...")
    vis_color_base = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_base, kptsB_base)
    cv2.imwrite(os.path.join(out_dir, "matches_color_coded_baseline.png"), vis_color_base)
    
    # Draw the weighted color-coded view
    print("Drawing Weighted Color-Coded Matches...")
    vis_color_weighted = draw_color_coded_matches(img_A_resized, img_B_resized, kptsA_weighted, kptsB_weighted)
    cv2.imwrite(os.path.join(out_dir, "matches_color_coded_weighted.png"), vis_color_weighted)

    print("Processing Baseline Visualization...")
    overlay_base, heatmap_base = create_overlay(cert_base_np, img_A_bgr)
    
    print("Processing Weighted Visualization...")
    overlay_weighted, heatmap_weighted = create_overlay(cert_weighted_np, img_A_bgr)
    
    # Compute a difference map (showing magnitude as a heatmap)
    diff = np.abs(cert_weighted_np - cert_base_np)
    # Amplify the difference for visualization (x5) and clip it
    diff_norm = np.clip(diff * 5, 0, 1)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)

    # Save output images
    cv2.imwrite(os.path.join(out_dir, "overlay_baseline.png"), overlay_base)
    cv2.imwrite(os.path.join(out_dir, "overlay_weighted.png"), overlay_weighted)
    cv2.imwrite(os.path.join(out_dir, "heatmap_baseline.png"), heatmap_base)
    cv2.imwrite(os.path.join(out_dir, "heatmap_weighted.png"), heatmap_weighted)
    cv2.imwrite(os.path.join(out_dir, "diff_heatmap_x5.png"), diff_heatmap)
    
    print(f"\nVisualizations saved to directory: {out_dir}")
    print(f"  - overlay_*.png: heatmap overlaid on the original image")
    print(f"  - heatmap_*.png: standalone heatmap")
    print(f"  - diff_heatmap_x5.png: difference heatmap")
    print(f"  - matches_*.png: match-line visualization")
    print(f"  - dense_points_*.png: dense matched-point distribution (left green, right red)")
    print(f"  - matches_color_coded_*.png: color-coded correspondence view (recommended)")

if __name__ == "__main__":
    main()