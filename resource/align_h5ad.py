import pandas as pd
import scanpy as sc
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ===== Input paths =====
CSV_PATH = "/public/home/shenninggroup/ivy/py_spatial/3omics/csv_ours/sm_transformed_coordinates_spomialign.csv"
H5AD_PATH = "/public/home/shenninggroup/ivy/py_spatial/3omics/h5ad_ours/adata_SM_with_spatialaligned.h5ad"
OUTPUT_H5AD = "/public/home/shenninggroup/ivy/py_spatial/3omics/h5ad_ours/adata_SM_with_spatialaligned.h5ad"

# 1. Read the CSV file
df = pd.read_csv(CSV_PATH)

# Keep only x_transformed and y_transformed
coords = df[["x_transformed", "y_transformed"]].to_numpy()

# 2. Read the existing h5ad file
adata = sc.read_h5ad(H5AD_PATH)

# 3. Store the coordinates in obsm['spatial']
adata.obsm["spatial"] = coords

# 4. Save the updated h5ad file
adata.write(OUTPUT_H5AD)

print(f"Written to obsm['spatial'] and saved to {OUTPUT_H5AD}")
