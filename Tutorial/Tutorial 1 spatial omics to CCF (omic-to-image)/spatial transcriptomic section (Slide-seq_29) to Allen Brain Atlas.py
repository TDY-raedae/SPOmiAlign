# # spatial transcriptomic section (Slide-seq_29) to Allen Brain Atlas
#
# This tutorial demonstrates omic-to-image alignment by registering the Slide-seq_29 spatial omics section to the Allen Brain Atlas image reference.
#
# Tutorial 1: spatial omics to CCF (omic-to-image)

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

from tutorial_utils import (
    generate_ssi_visualization,
    get_tutorial_paths,
    run_omic_to_image_alignment,
    show_images,
)

DATA_DIR, OUTPUT_ROOT = get_tutorial_paths(PROJECT_ROOT)

# ## 2. Parameter settings
#
# | Parameter | Meaning |
# | --- | --- |
# | `SAMPLE_ID` | Output folder name under `output/h5ad_2_img/`. |
# | `SOURCE_OMIC_PATH` | Source spatial omics h5ad path in data. |
# | `TARGET_IMAGE_PATH` | Reference image path in data. |
# | `SSI_IMAGE_PATH` | The path of rendered SSI image. |
# | `manual_rotate` | Clockwise manual rotation applied when rendering the source h5ad into SSI. |
# | `SSI_dpi` | SSI rendering resolution. The default value is 150. |
# | `x_coordinate` / `y_coordinate` | Spot coordinate columns used for SSI rendering. |
# | `SPOT_UMI` | UMI is calculated by default; use this h5ad obs key if available. |
# | `threshold_percentile` | Optional intensity percentile filter; `None` keeps all valid spots. |
# | `SPOT` | Spot shape (`square` / `circle`) and visualization radius. |
# | `Alignment_mode` | SPOmiAlign supports Rigid, Non-Rigid, and Affine alignment modes. |

SAMPLE_ID = "PUCK29"
SOURCE_OMIC_PATH = "Tutorial 1 spatial omics to CCF (omic-to-image)/Puck_Num_29.h5ad"
TARGET_IMAGE_PATH = "Tutorial 1 spatial omics to CCF (omic-to-image)/CCF_100048576_205.png"
SSI_IMAGE_PATH = None

SSI_PARAMS = {
    "manual_rotate": 180,
    "SSI_dpi": 150,
    "x_coordinate": "Raw_Slideseq_X",
    "y_coordinate": "Raw_Slideseq_Y",
    "SPOT_UMI": "nFeature_Spatial",
    "threshold_percentile": None,
    "SPOT": {"shape": "circle", "radius": 5},
}

ALIGNMENT_PARAMS = {
    "Alignment_mode": "affine+bspline",
}
SAMPLE_ID, SOURCE_OMIC_PATH, TARGET_IMAGE_PATH, SSI_IMAGE_PATH, SSI_PARAMS, ALIGNMENT_PARAMS

# ## 3. Generate SSI image
#
# The SSI image is generated from the spatial omics h5ad before alignment. For these Slide-seq examples, spot coordinates are read from `Raw_Slideseq_X` and `Raw_Slideseq_Y`, and spot intensity is read from `nFeature_Spatial`.

ssi = generate_ssi_visualization(
    data_root=DATA_DIR,
    output_root=OUTPUT_ROOT,
    sample_id=SAMPLE_ID,
    source_omic_path=SOURCE_OMIC_PATH,
    **SSI_PARAMS,
)

show_images([ssi["source_section_visualization"]], ["Generated SSI"], figsize=(6, 6))
ssi

# ## 4. Run SPOmiAlign

result = run_omic_to_image_alignment(
    data_root=DATA_DIR,
    output_root=OUTPUT_ROOT,
    sample_id=SAMPLE_ID,
    source_omic_path=SOURCE_OMIC_PATH,
    target_image_path=TARGET_IMAGE_PATH,
    SSI_IMAGE_PATH=SSI_IMAGE_PATH,
    **SSI_PARAMS,
    **ALIGNMENT_PARAMS,
)

# ## 5. Inspect outputs

show_images(
    [result["source_section_visualization"], result["target_section_visualization"], result["overlay"]],
    ["Generated SSI", "CCF reference", "Aligned slice overlay"],
    figsize=(18, 6),
)
result
