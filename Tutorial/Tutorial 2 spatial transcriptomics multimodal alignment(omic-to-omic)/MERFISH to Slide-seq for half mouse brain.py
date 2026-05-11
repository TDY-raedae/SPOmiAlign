# # MERFISH to Slide-seq for half mouse brain
#
# Tutorial 2: spatial transcriptomics multimodal alignment (omic-to-omic)

# ## 1. Load package and data paths

from pathlib import Path
import sys

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

from tutorial_utils import get_tutorial_paths, run_image_to_image_alignment, show_images

DATA_DIR, OUTPUT_ROOT = get_tutorial_paths(PROJECT_ROOT)

# ## 2. Parameter settings
#
# | Parameter | Meaning |
# | --- | --- |
# | `SAMPLE_ID` | Output folder name under `output/img_2_img/`. |
# | `target_image_path` | Fixed reference image path in data. |
# | `source_image_path` | Moving image path in data. |
# | `Alignment_mode` | SPOmiAlign supports Rigid, Non-Rigid, and Affine alignment modes. |

PARAMS = {
    "sample_id": "092_to_Puck57",
    "target_image_path": "Tutorial 2 spatial transcriptomics multimodal alignment(omic-to-omic)/Puck_Num_57.png",
    "source_image_path": "Tutorial 2 spatial transcriptomics multimodal alignment(omic-to-omic)/092.png",
    "Alignment_mode": "affine+bspline",
}
PARAMS

# ## 3. Run SPOmiAlign

result = run_image_to_image_alignment(
    data_root=DATA_DIR,
    output_root=OUTPUT_ROOT,
    **PARAMS,
)

# ## 4. Inspect outputs

show_images(
    [
        result["target_section_visualization"],
        result["source_section_visualization"],
        result["before_overlay"],
        result["aligned_source"],
        result["after_overlay"],
    ],
    ["Target image", "Source image", "Transparent overlay before alignment", "Aligned source", "Transparent overlay after alignment"],
    figsize=(24, 5),
)
result
