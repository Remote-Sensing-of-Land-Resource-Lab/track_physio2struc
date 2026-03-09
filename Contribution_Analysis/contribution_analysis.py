# ==========================================================
# Random forest: global model using all time-step data
# NO SAMPLING VERSION
# X = [SIFratio, LAI_anoma], y = gpp_anomaly
# ==========================================================

import xarray as xr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

# ------------------------------
# 1. Load data
# ------------------------------
gpp = xr.open_dataarray("gs_drought_gosif_gpp_raw_anomaly.nc")
sif = xr.open_dataarray("SIFratio.nc")
lai = xr.open_dataarray("lai_anoma.nc")
swc = xr.open_dataset("gs_drought_swc_8day_2001_2022.nc")["swc"]
thr = xr.open_dataset("threshold_metrics_8daymin_gosif_gpp.nc")["thr_final_second"] / 100

# align threshold
thr = thr.broadcast_like(gpp.isel(time=0))

# ------------------------------
# 2. only keep pixels with threshold
# ------------------------------
valid = np.isfinite(thr)

gpp = gpp.where(valid)
sif = sif.where(valid)
lai = lai.where(valid)
swc = swc.where(valid)
thr = thr.where(valid)

nt, ny, nx = gpp.shape

# ==========================================================
# Use ALL pixels (no spatial sampling)
# ==========================================================

valid_indices = np.argwhere(valid.values)

selected_j = valid_indices[:,0]
selected_i = valid_indices[:,1]

print("Total valid pixels:", len(selected_j))


# ==========================================================
# Extract all time-step samples
# ==========================================================

X_above = []
y_above = []
X_below = []
y_below = []

print("Extracting time-step samples...")

for j, i in tqdm(zip(selected_j, selected_i), total=len(selected_j)):

    tthr = thr[j, i].item()
    if np.isnan(tthr):
        continue

    swc_ts = swc[:, j, i].values
    sif_ts = sif[:, j, i].values
    lai_ts = lai[:, j, i].values
    gpp_ts = gpp[:, j, i].values

    # pixel-level centering
    sif_ts = sif_ts - np.nanmean(sif_ts)
    lai_ts = lai_ts - np.nanmean(lai_ts)
    gpp_ts = gpp_ts - np.nanmean(gpp_ts)

    mask_above = (swc_ts > tthr) & np.isfinite(sif_ts) & np.isfinite(lai_ts) & np.isfinite(gpp_ts)
    mask_below = (swc_ts <= tthr) & np.isfinite(sif_ts) & np.isfinite(lai_ts) & np.isfinite(gpp_ts)

    if np.any(mask_above):
        X_above.extend(np.column_stack([sif_ts[mask_above], lai_ts[mask_above]]))
        y_above.extend(gpp_ts[mask_above])

    if np.any(mask_below):
        X_below.extend(np.column_stack([sif_ts[mask_below], lai_ts[mask_below]]))
        y_below.extend(gpp_ts[mask_below])


# ------------------------------
# Build DataFrame
# ------------------------------
df_above = pd.DataFrame(X_above, columns=["sif", "lai"])
df_above["gpp"] = y_above

df_below = pd.DataFrame(X_below, columns=["sif", "lai"])
df_below["gpp"] = y_below

print("Samples above:", len(df_above))
print("Samples below:", len(df_below))


# ------------------------------
# Train RF
# ------------------------------
def train_rf(df):

    X = df[["sif", "lai"]]
    y = df["gpp"]

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=0
    )

    rf.fit(X, y)

    ypred = rf.predict(X)
    r2 = r2_score(y, ypred)

    return rf, r2


rf_above, r2_above = train_rf(df_above)
rf_below, r2_below = train_rf(df_below)


# ------------------------------
# Output
# ------------------------------
print("\n===== R² RESULTS =====")
print("Above threshold:", round(r2_above, 3))
print("Below threshold:", round(r2_below, 3))

print("\n===== Feature importance (ABOVE) =====")
for name, val in zip(["sif", "lai"], rf_above.feature_importances_):
    print(f"{name}: {val:.3f}")

print("\n===== Feature importance (BELOW) =====")
for name, val in zip(["sif", "lai"], rf_below.feature_importances_):
    print(f"{name}: {val:.3f}")