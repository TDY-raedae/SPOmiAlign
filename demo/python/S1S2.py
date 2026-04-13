import os
import sys

sys.path.append("../../SPOmiAlign")

from roma import align_and_process_images


# =========================
# Configure paths
# =========================
DATA_DIR = "../../SPOmiAlign_Repro"
img1_path = os.path.join(DATA_DIR, "output_image", "E15_5-S1-HE.jpg")  # Target (ST)
img2_path = os.path.join(DATA_DIR, "output_image", "E15_5-S2-HE_warped_rt15.png")  # Source (SM)

SAVE_DIR = "../../output"
SAVE_PATH = os.path.join(SAVE_DIR, "img_2_img", "S2toS1")
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Working directory is ready: {os.path.abspath(SAVE_PATH)}")

# =========================
# Validate input files
# =========================
files_to_check = {
    "Target image": img1_path,
    "Source image": img2_path,
}

print("\nChecking input files...")
missing_files = []
for name, path in files_to_check.items():
    if os.path.exists(path):
        file_size = os.path.getsize(path) / (1024 * 1024)
        print(f"[OK] {name} found: {os.path.basename(path)} ({file_size:.2f} MB)")
    else:
        print(f"[ERROR] {name} not found: {os.path.abspath(path)}")
        missing_files.append(path)

if missing_files:
    sys.exit("[ERROR] Program stopped: required input files are missing.")
else:
    print("All files are ready. Starting the workflow.\n" + "-" * 30)


# =========================
# Step 1: Image alignment
# =========================
print("Step 1: Align the source image with the target image.\n" + "-" * 30)
save_path_alignment = os.path.join(SAVE_PATH, "alignment")

align_and_process_images(
    img1_path=img1_path,
    img2_path=img2_path,
    method="affine+bspline",
    output_dir=save_path_alignment,
    rotate=0.0,
    scale=1.0,
)

print("\nWorkflow completed successfully.")
