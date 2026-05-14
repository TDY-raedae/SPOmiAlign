# Auto-generated from the matching notebook.


# # spatial multi-omics alignment with paired images
#
# Tutorial 4: spatial multi-omics alignment with paired images (image-to-image)

# ## 1. Load package and data paths

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import cv2
try:
    START_DIR = Path(__file__).resolve().parent
except NameError:
    START_DIR = Path.cwd()

PROJECT_ROOT = next(
    candidate for candidate in (START_DIR, *START_DIR.parents)
    if (candidate / "SPOmiAlign").is_dir()
)
spomialign_path = PROJECT_ROOT / "SPOmiAlign"
if str(spomialign_path) not in sys.path:
    sys.path.insert(0, str(spomialign_path))

from tutorial_utils import get_tutorial_paths, run_image_to_image_alignment, read_bgr

DATA_DIR, OUTPUT_ROOT = get_tutorial_paths(PROJECT_ROOT)
# %matplotlib inline

# ## 2. Parameter settings
#
# | Parameter | Meaning |
# | --- | --- |
# | `SAMPLE_ID` | Output folder name under `output/img_2_img/`. |
# | `target_image_path` | Fixed reference image path in data. |
# | `source_image_path` | Moving image path in data. |
# | `target_h5ad_path` | Target spatial omics file already mapped to the reference image frame. |
# | `source_h5ad_path` | Source spatial omics file to transform with the image alignment. |
# | `target_h5ad_spatial_key` / `source_h5ad_spatial_key` | Coordinate key used by the corresponding h5ad file. |
# | `Alignment_mode` | SPOmiAlign supports three alignment modes: Rigid (`Rigid`), Affine (`Affine`, `Homography`), and Non-Rigid (`bspline`, `affine+bspline`). |
# | `device` | Torch device used by alignment, for example `cuda:0`, `cuda:1`, or `cpu`; `None` keeps automatic selection. |

PARAMS = {
    "sample_id": "S2toS1",
    "target_image_path": "Tutorial 4 spatial multi-omics alignment with paired images(image-to-image)/E15_5-S1-HE.jpg",
    "source_image_path": "Tutorial 4 spatial multi-omics alignment with paired images(image-to-image)/E15_5-S2-HE_warped_rt15.png",
    "target_h5ad_path": "Tutorial 4 spatial multi-omics alignment with paired images(image-to-image)/s1_rna_mapped.h5ad",
    "source_h5ad_path": "Tutorial 4 spatial multi-omics alignment with paired images(image-to-image)/s2_motif_unaligned.h5ad",
    "target_h5ad_spatial_key": "spatial",
    "source_h5ad_spatial_key": "spatial",
    "Alignment_mode": "affine+bspline",
    "device": "cuda:1",
}
PARAMS

# ## 3. Run SPOmiAlign

result = run_image_to_image_alignment(
    data_root=DATA_DIR,
    output_root=OUTPUT_ROOT,
    **PARAMS,
)

# ## 4. Outputs

images=[result["target_section_visualization"], result["source_section_visualization"], result["before_overlay"], result["aligned_source"], result["after_overlay"]]
titles=["Target image", "Source image", "Overlay before alignment", "Aligned source", "Overlay after alignment"]
figsize=(24, 5)
titles = titles or ["" for _ in images]
if figsize is None:
    figsize = (5.5 * len(images), 5.5)
fig, axes = plt.subplots(1, len(images), figsize=figsize)
if len(images) == 1:
    axes = [axes]
for ax, image_or_path, title in zip(axes, images, titles):
    image = read_bgr(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

images=[
    result["target_h5ad_visualization"],
    result["transformed_h5ad_visualization"],
    result["h5ad_overlay"],
]
titles=["Target h5ad (`spatial`)", "Aligned source h5ad (`spatial`)", "Aligned source h5ad on target image"]
figsize=(15, 5)
titles = titles or ["" for _ in images]
if figsize is None:
    figsize = (5.5 * len(images), 5.5)
fig, axes = plt.subplots(1, len(images), figsize=figsize)
if len(images) == 1:
    axes = [axes]
for ax, image_or_path, title in zip(axes, images, titles):
    image = read_bgr(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

result["transformed_h5ad"]
