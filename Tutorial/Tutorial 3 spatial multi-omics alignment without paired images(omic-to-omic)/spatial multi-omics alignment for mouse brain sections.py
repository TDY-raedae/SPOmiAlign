# # spatial multi-omics alignment for mouse brain sections
#
# Tutorial 3: spatial multi-omics alignment without paired images (omic-to-omic)

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
    run_omic_to_omic_alignment,
    run_reassignment,
    show_images,
)

DATA_DIR, OUTPUT_ROOT = get_tutorial_paths(PROJECT_ROOT)

# ## 2. Parameter settings
#
# | Parameter | Meaning |
# | --- | --- |
# | `target_section_path` | reference ST h5ad paths in `Data/`. |
# | `source_section_path` | source SM h5ad paths relative to `Data/`. |
# | `target_rotate` / `source_rotate` | Clockwise manual rotation applied when rendering the target/source h5ad into SSI. |
# | `SPOT` | Spot shape (`square` / `circle`) and visualization radius. |
# | `Alignment_mode` | SPOmiAlign supports Rigid, Non-Rigid, and Affine alignment modes. |

DATA_PARAMS = {
    "target_section_path": 'Tutorial 3 spatial multi-omics alignment without paired images(omic-to-omic)/st_withIntensity.h5ad',
    "source_section_path": 'Tutorial 3 spatial multi-omics alignment without paired images(omic-to-omic)/sm_withIntensity.h5ad',
}

SSI_PARAMS = {
    "target_rotate": 0.0,
    "source_rotate": 60.0,
    "SPOT": {"shape": "square", "target_radius": 15, "source_radius": 12},
}

ALIGNMENT_PARAMS = {
    "sample_id": "sm2st1",
    "Alignment_mode": "affine+bspline",
}
DATA_PARAMS, SSI_PARAMS, ALIGNMENT_PARAMS

# ## 3. Generate SSI images
#
# The target and source h5ad files are rendered as SSI images before matching. Coordinates are read from `obsm['spatial']`, and spot intensity is computed from expression counts in `X` by default.

target_h5ad = DATA_DIR / DATA_PARAMS["target_section_path"]
source_h5ad = DATA_DIR / DATA_PARAMS["source_section_path"]

target_ssi = generate_ssi_visualization(
    output_root=OUTPUT_ROOT,
    sample_id=ALIGNMENT_PARAMS["sample_id"],
    h5ad_path=target_h5ad,
    label="target/ST",
    output_name="st_section_visualization.png",
    rotate=SSI_PARAMS["target_rotate"],
    SPOT={"shape": SSI_PARAMS["SPOT"]["shape"], "radius": SSI_PARAMS["SPOT"]["target_radius"]},
)
source_ssi = generate_ssi_visualization(
    output_root=OUTPUT_ROOT,
    sample_id=ALIGNMENT_PARAMS["sample_id"],
    h5ad_path=source_h5ad,
    label="source/SM",
    output_name="sm_section_visualization.png",
    rotate=SSI_PARAMS["source_rotate"],
    SPOT={"shape": SSI_PARAMS["SPOT"]["shape"], "radius": SSI_PARAMS["SPOT"]["source_radius"]},
)

ssi = {
    "target_section_visualization": target_ssi["section_visualization"],
    "source_section_visualization": source_ssi["section_visualization"],
    "target_coord_config": target_ssi["coord_config"],
    "source_coord_config": source_ssi["coord_config"],
    "target_render_meta": target_ssi["render_meta"],
    "source_render_meta": source_ssi["render_meta"],
    "target_render_style": target_ssi["render_style"],
    "source_render_style": source_ssi["render_style"],
    "source_origin": source_ssi["origin"],
}

show_images(
    [ssi["target_section_visualization"], ssi["source_section_visualization"]],
    ["Target SSI", "Source SSI"],
    figsize=(12, 6),
)
ssi

# ## 4. Run SPOmiAlign alignment

result = run_omic_to_omic_alignment(
    output_root=OUTPUT_ROOT,
    target_h5ad=target_h5ad,
    source_h5ad=source_h5ad,
    run_reassignment=False,
    **SSI_PARAMS,
    **ALIGNMENT_PARAMS,
)

# ## 5. Reassignment module
#
# | Parameter | Meaning |
# | --- | --- |
# | `direction` | Reassignment direction; default is bidirectional, otherwise along this direction. |
# | `s1_spatial_key` | Aligned SM coordinates produced by SPOmiAlign. |
# | `s2_spatial_key` | Original ST coordinates used as the reference layout. |

REASSIGNMENT_PARAMS = {
    "direction": None,
    "s1_spatial_key": "spatial_spomialign",
    "s2_spatial_key": "spatial_raw",
}

result.update(
    run_reassignment(
        output_root=OUTPUT_ROOT,
        sample_id=ALIGNMENT_PARAMS["sample_id"],
        target_h5ad=target_h5ad,
        source_h5ad=source_h5ad,
        aligned_source_h5ad=result["transformed_h5ad"],
        **REASSIGNMENT_PARAMS,
    )
)
result["high_to_low"], result["low_to_high"]

# ## 6. Inspect outputs

show_images(
    [result["target_section_visualization"], result["source_section_visualization"], result["before_overlay"], result["after_overlay"]],
    ["Reference ST SSI", "Source SM SSI", "Transparent overlay before alignment", "Transparent overlay after alignment"],
    figsize=(22, 5),
)

show_images(
    [
        result["high_to_low"]["plot_dir"] / "S1_umi.png",
        result["high_to_low"]["plot_dir"] / "S2_umi.png",
        result["high_to_low"]["plot_dir"] / "reassigned_high_to_low_umi.png",
    ],
    ["Aligned SM", "Reference ST", "High-to-low reassignment"],
    figsize=(18, 5),
)
result
