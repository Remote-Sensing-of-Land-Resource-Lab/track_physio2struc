import xarray as xr
import numpy as np

# 1. 读取文件
ds = xr.open_dataset('threshold_metrics_zscore.nc')

# 2. 对 thr_final 和 thr_final_second 应用逻辑
ds["thr_final"] = ds["thr_final"].where(
    (ds["thr_final"] > ds["sm_binned_min"]) & (ds["thr_final"] < ds["sm_binned_max"])
)

ds["thr_final_second"] = ds["thr_final_second"].where(
    (ds["thr_final_second"] > ds["sm_binned_min_second"]) & (ds["thr_final_second"] < ds["sm_binned_max_second"])
)

# 3. 覆盖保存（或写入新文件）
ds.to_netcdf("threshold_metrics_zscore_fixed.nc", mode="w")

print("✅ 已修正边界处的 thr_final / thr_final_second 并保存为新文件！")

