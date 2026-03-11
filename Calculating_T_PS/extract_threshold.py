# -*- coding: utf-8 -*-
"""
Compute GPP-SM threshold metrics for each grid cell.

GPP and SM can be download from
https://drive.google.com/drive/folders/1A9oUD0ZpZ8lYN6qbZozZl6p3sm0fNEz6?usp=sharing

Please renew your file locations and variable names below if needed.
"""

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter
from numpy.polynomial.polyutils import RankWarning


# ==========================================================
# 1. User settings
#    Modify these paths / names / parameters before running
# ==========================================================
GPP_PATH = "gs_drought_gpp_1982_2013.nc"
SM_PATH = "gs_drought_sm_1982_2013.nc"
OUTPUT_PATH = "threshold_metrics_1982_2013.nc"

# If set to None, the script will try to detect the variable automatically
GPP_VAR = None
SM_VAR = None

# Multiply SM by this factor if you want to convert from m3/m3 to percentage-like values
SM_SCALE_FACTOR = 100.0

# Chunk settings for dask/xarray
TIME_CHUNK = -1
LAT_CHUNK = 128
LON_CHUNK = 128

# Threshold fitting settings
USE_SG_FILTER = False
MIN_PER_BIN = 1
MIN_BINS = 6
BIN_WIDTH = 1.0
SG_WINDOW = 5
SG_POLYORDER = 2
DEBUG = False

# Output compression settings
ZLIB = True
COMPLEVEL = 4
OUTPUT_DTYPE = np.float32

# Dask scheduler
DASK_SCHEDULER = "threads"


# ==========================================================
# 2. Silence fitting / numerical warnings
# ==========================================================
warnings.filterwarnings("ignore", category=RankWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================================
# 3. Helper functions for reading input data
# ==========================================================
def infer_data_var(ds: xr.Dataset, preferred_name: str = None) -> str:
    """
    Infer the target variable name from a Dataset.

    Priority:
    1. Use preferred_name if it exists in the dataset
    2. If there is only one data variable, use it
    3. Try several common variable names
    """
    if preferred_name is not None and preferred_name in ds.data_vars:
        return preferred_name

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    common_candidates = [
        "GPP", "gpp",
        "sm", "SM",
        "swc", "SWC",
        "soil_moisture"
    ]
    for name in common_candidates:
        if name in ds.data_vars:
            return name

    raise ValueError(
        f"Cannot infer variable name automatically. "
        f"Available variables are: {data_vars}. "
        f"Please set GPP_VAR or SM_VAR manually."
    )


def load_dataarray(nc_path: str, var_name: str = None) -> xr.DataArray:
    """
    Open a NetCDF file and return the selected DataArray.
    """
    ds = xr.open_dataset(nc_path, chunks={})
    target_var = infer_data_var(ds, preferred_name=var_name)
    return ds[target_var]


def validate_and_align_time(gpp: xr.DataArray, sm: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Make sure GPP and SM have the same time length and
    directly assign GPP time coordinate to SM, following your original workflow.
    """
    if "time" not in gpp.dims or "time" not in sm.dims:
        raise ValueError("Both input datasets must contain 'time' dimension.")

    if gpp.sizes["time"] != sm.sizes["time"]:
        raise ValueError(
            f"GPP and SM time lengths differ: "
            f"{gpp.sizes['time']} vs {sm.sizes['time']}"
        )

    sm = sm.assign_coords(time=gpp.time)
    return gpp, sm


# ==========================================================
# 4. Basic fitting statistics
# ==========================================================
def _r2_rmse(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """
    Calculate R2 and RMSE between observed and fitted values.
    """
    resid = y - yhat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(y))
    return r2, rmse


# ==========================================================
# 5. Segmented model definitions
# ==========================================================
def seg1_func(x: np.ndarray, a1: float, b_ratio: float, x0: float, c: float) -> np.ndarray:
    """
    One-break segmented linear model.

    Before x0: slope = a1
    After  x0: slope = a1 * b_ratio
    """
    b = a1 * b_ratio
    return np.where(x < x0, a1 * x + c, b * (x - x0) + (a1 * x0 + c))


def seg2_func(
    x: np.ndarray,
    a1: float,
    b1_ratio: float,
    x1: float,
    b2_ratio: float,
    x2: float,
    c: float
) -> np.ndarray:
    """
    Two-break segmented linear model.

    Segment 1 slope = a1
    Segment 2 slope = a1 * b1_ratio
    Segment 3 slope = a1 * b1_ratio * b2_ratio
    """
    a2 = a1 * b1_ratio
    a3 = a2 * b2_ratio

    y1 = a1 * x + c
    y2 = a2 * (x - x1) + (a1 * x1 + c)
    y3 = a3 * (x - x2) + (a2 * (x2 - x1) + (a1 * x1 + c))

    return np.where(x < x1, y1, np.where(x < x2, y2, y3))


# ==========================================================
# 6. Binning function
# ==========================================================
def _bin_trim_with_counts(
    sm: np.ndarray,
    gpp: np.ndarray,
    min_per_bin: int,
    min_bins: int,
    bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin SM values, then compute mean SM and mean GPP within each bin.

    Steps:
    1. Remove NaN pairs
    2. Build bins along SM
    3. Keep bins with enough samples
    4. Remove extreme GPP values outside [-100, 100]
    5. Return sorted binned SM, binned GPP, and counts
    """
    m0 = np.isfinite(sm) & np.isfinite(gpp)
    sm0, gpp0 = sm[m0], gpp[m0]

    if sm0.size < min_per_bin * min_bins:
        return None, None, None

    lo, hi = np.floor(np.nanmin(sm0)), np.ceil(np.nanmax(sm0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None, None, None

    edges = np.arange(lo, hi + bin_width, bin_width)
    if edges.size < 3:
        return None, None, None

    x_list, y_list, n_list = [], [], []
    bin_id = np.digitize(sm0, edges) - 1

    for bid in np.unique(bin_id):
        if bid < 0 or bid >= len(edges) - 1:
            continue

        sel = bin_id == bid
        if sel.sum() < min_per_bin:
            continue

        xx, yy = sm0[sel], gpp0[sel]

        # Exclude extremely large/small GPP anomaly values
        keep = (yy <= 100) & (yy >= -100)
        if keep.sum() < min_per_bin:
            continue

        x_list.append(xx[keep].mean())
        y_list.append(yy[keep].mean())
        n_list.append(keep.sum())

    if len(x_list) < min_bins:
        return None, None, None

    x = np.asarray(x_list)
    y = np.asarray(y_list)
    n = np.asarray(n_list)

    order = np.argsort(x)
    return x[order], y[order], n[order]


# ==========================================================
# 7. Model fitting functions
# ==========================================================
def _fit_poly(x: np.ndarray, y: np.ndarray, deg: int, enforce_positive: bool = False) -> tuple[float, float, float]:
    """
    Fit a polynomial model and return:
    rmse, r2, slope

    For quadratic fit, the returned 'slope' is only kept for interface consistency;
    it is not used later.
    """
    try:
        p = np.polyfit(x, y, deg)
        yhat = np.polyval(p, x)
        r2, rmse = _r2_rmse(y, yhat)
        slope = p[-2] if deg >= 1 else np.nan

        # For linear models in this workflow, negative slope is not accepted
        if enforce_positive and slope <= 0:
            r2 = -1
            rmse = np.nan

        return rmse, r2, slope
    except Exception:
        return np.nan, np.nan, np.nan


def _fit_seg1(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """
    Fit one-break segmented model and return:
    rmse, r2, slope_before_break, slope_after_break, threshold, sm_min, sm_max
    """
    try:
        sm_min, sm_max = float(np.min(x)), float(np.max(x))
        slope_init = max((y[-1] - y[0]) / max(x[-1] - x[0], 1e-12), 1e-6)
        x0_init = float(np.median(x))
        c_init = float(np.interp(x0_init, x, y))

        p0 = [slope_init, 0.5, x0_init, c_init]
        bounds = ([0, 0, sm_min, -np.inf], [np.inf, 1.0, sm_max, np.inf])

        popt, _ = curve_fit(seg1_func, x, y, p0=p0, bounds=bounds, maxfev=2000)
        a1, b_ratio, x0, c = popt
        a2 = a1 * b_ratio

        yhat = seg1_func(x, *popt)
        r2, rmse = _r2_rmse(y, yhat)

        return rmse, r2, a1, a2, x0, sm_min, sm_max
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def _fit_seg2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Fit two-break segmented model and return:
    rmse, r2, slope1, slope2, slope3, threshold1, threshold2, sm_min, sm_max
    """
    try:
        sm_min, sm_max = float(np.min(x)), float(np.max(x))
        slope_init = max((y[-1] - y[0]) / max(x[-1] - x[0], 1e-12), 1e-6)
        x1_init = x[int(len(x) * 0.4)]
        x2_init = x[int(len(x) * 0.8)]
        c_init = float(np.interp(x1_init, x, y))

        p0 = [slope_init, 0.5, x1_init, 0.5, x2_init, c_init]
        bounds = (
            [0, 0, sm_min, 0, sm_min, -np.inf],
            [np.inf, 1.0, sm_max, 1.0, sm_max, np.inf]
        )

        popt, _ = curve_fit(seg2_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        a1, b1_ratio, x1, b2_ratio, x2, c = popt
        a2 = a1 * b1_ratio
        a3 = a2 * b2_ratio

        yhat = seg2_func(x, *popt)
        r2, rmse = _r2_rmse(y, yhat)

        return rmse, r2, a1, a2, a3, x1, x2, sm_min, sm_max
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


# ==========================================================
# 8. Core threshold calculation for one pixel
# ==========================================================
def compute_threshold(
    gpp_ts,
    sm_ts,
    *,
    use_sg_filter=False,
    min_per_bin=1,
    min_bins=6,
    bin_width=1.0,
    sg_window=5,
    sg_polyorder=2,
    debug=False
):
    """
    Compute GPP-SM threshold metrics for one pixel time series.

    This function calculates two sets of metrics:
    1. start -> max:
       fit only from the beginning of the binned curve to the peak GPP bin (the result we used)
    2. start -> end:
       fit using the full binned curve (just to see how result would be different if we did not exclude GPP increase under moderate drought in very wet regions)

    Returned outputs (28 total):
    ----------------------------------------------------------
    start -> max:
        rmse_linear_smax
        r2_linear_smax
        slope_linear_smax
        rmse_quadratic_smax
        r2_quadratic_smax
        rmse_segmented_smax
        r2_segmented_smax
        seg_slope_low_smax
        seg_slope_high_smax
        seg_threshold_smax
        sm_binned_min_smax
        sm_binned_max_smax
        thr_final_smax

    start -> end:
        rmse_linear_send
        r2_linear_send
        slope_linear_send
        rmse_quadratic_send
        r2_quadratic_send
        rmse_segmented_send
        r2_segmented_send
        seg_slope_low_send
        seg_slope_high_send
        seg_threshold_send
        sm_binned_min_send
        sm_binned_max_send
        thr_final_send

    counts:
        n_valid_smax
        n_valid_send
    """
    gpp = np.asarray(gpp_ts, float)
    sm = np.asarray(sm_ts, float)

    # Step 1: binning
    xb, yb, nb = _bin_trim_with_counts(
        sm=sm,
        gpp=gpp,
        min_per_bin=min_per_bin,
        min_bins=min_bins,
        bin_width=bin_width
    )

    if xb is None:
        return (np.nan,) * 28

    # Number of valid binned points used in each fitting domain
    n_valid_smax = len(xb[: int(np.argmax(yb)) + 1])
    n_valid_send = len(xb)

    # Optional smoothing only for identifying the peak location
    if use_sg_filter and len(yb) >= 5:
        win = sg_window if sg_window % 2 == 1 else sg_window + 1
        win = min(win, len(yb))
        yb_s = savgol_filter(yb, win, min(sg_polyorder, win - 1))
    else:
        yb_s = yb.copy()

    i_max = int(np.argmax(yb_s))

    # Two fitting domains
    xb_smax, yb_smax = xb[: i_max + 1], yb[: i_max + 1]
    xb_send, yb_send = xb, yb

    # ==========================================================
    # start -> max
    # ==========================================================
    rmse_lin_smax, r2_lin_smax, slope_lin_smax = _fit_poly(
        xb_smax, yb_smax, deg=1, enforce_positive=True
    )
    rmse_quad_smax, r2_quad_smax, _ = _fit_poly(
        xb_smax, yb_smax, deg=2
    )
    (
        rmse_seg_smax,
        r2_seg_smax,
        s_low_smax,
        s_high_smax,
        thr_smax,
        sm_min_smax,
        sm_max_smax,
    ) = _fit_seg1(xb_smax, yb_smax)

    # Keep threshold only when it lies strictly inside the binned SM range
    thr_final_smax = (
        thr_smax
        if (
            np.isfinite(thr_smax)
            and np.isfinite(sm_min_smax)
            and np.isfinite(sm_max_smax)
            and (thr_smax > sm_min_smax)
            and (thr_smax < sm_max_smax)
        )
        else np.nan
    )

    # ==========================================================
    # start -> end
    # ==========================================================
    rmse_lin_send, r2_lin_send, slope_lin_send = _fit_poly(
        xb_send, yb_send, deg=1, enforce_positive=True
    )
    rmse_quad_send, r2_quad_send, _ = _fit_poly(
        xb_send, yb_send, deg=2
    )

    rmse_seg_send = np.nan
    r2_seg_send = np.nan
    s_low_send = np.nan
    s_high_send = np.nan
    thr_send = np.nan
    sm_min_send = np.nan
    sm_max_send = np.nan

    # If the last 4 binned points are all decreasing,
    # allow a more flexible two-break segmented fit
    is_decreasing = len(yb_send) >= 4 and np.all(np.diff(yb_send[-4:]) < 0)

    if is_decreasing:
        try:
            (
                rmse_seg_send,
                r2_seg_send,
                s1_send,
                s2_send,
                s3_send,
                thr1_send,
                thr2_send,
                sm_min_send,
                sm_max_send,
            ) = _fit_seg2(xb_send, yb_send)

            # Keep the first break as the main threshold
            thr_send = thr1_send
            s_low_send = s1_send
            s_high_send = s2_send
        except Exception:
            pass
    else:
        try:
            (
                rmse_seg_send,
                r2_seg_send,
                s_low_send,
                s_high_send,
                thr_send,
                sm_min_send,
                sm_max_send,
            ) = _fit_seg1(xb_send, yb_send)
        except Exception:
            pass

    # Keep threshold only when it lies within the binned SM range
    thr_final_send = (
        thr_send
        if (
            np.isfinite(thr_send)
            and np.isfinite(sm_min_send)
            and np.isfinite(sm_max_send)
            and (thr_send >= sm_min_send)
            and (thr_send <= sm_max_send)
        )
        else np.nan
    )

    return (
        # start -> max
        rmse_lin_smax, r2_lin_smax, slope_lin_smax,
        rmse_quad_smax, r2_quad_smax,
        rmse_seg_smax, r2_seg_smax,
        s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax, thr_final_smax,

        # start -> end
        rmse_lin_send, r2_lin_send, slope_lin_send,
        rmse_quad_send, r2_quad_send,
        rmse_seg_send, r2_seg_send,
        s_low_send, s_high_send, thr_send, sm_min_send, sm_max_send, thr_final_send,

        # counts
        n_valid_smax, n_valid_send
    )


# ==========================================================
# 9. Build output dataset
# ==========================================================
def build_output_dataset(gpp: xr.DataArray, out: tuple) -> xr.Dataset:
    """
    Convert the tuple output from apply_ufunc into a labeled xarray Dataset.
    """
    (
        rmse_lin_smax, r2_lin_smax, slope_lin_smax,
        rmse_quad_smax, r2_quad_smax,
        rmse_seg_smax, r2_seg_smax,
        s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax, thr_final_smax,

        rmse_lin_send, r2_lin_send, slope_lin_send,
        rmse_quad_send, r2_quad_send,
        rmse_seg_send, r2_seg_send,
        s_low_send, s_high_send, thr_send, sm_min_send, sm_max_send, thr_final_send,

        n_valid_smax, n_valid_send
    ) = out

    ds_all = xr.Dataset(
        data_vars=dict(
            # Primary results: start -> end
            rmse_linear=rmse_lin_send,
            r2_linear=r2_lin_send,
            slope_linear=slope_lin_send,
            rmse_quadratic=rmse_quad_send,
            r2_quadratic=r2_quad_send,
            rmse_segmented=rmse_seg_send,
            r2_segmented=r2_seg_send,
            seg_slope_low=s_low_send,
            seg_slope_high=s_high_send,
            seg_threshold=thr_send,
            sm_binned_min=sm_min_send,
            sm_binned_max=sm_max_send,
            thr_final=thr_final_send,
            n_valid=n_valid_send,

            # Secondary results: start -> max
            rmse_linear_second=rmse_lin_smax,
            r2_linear_second=r2_lin_smax,
            slope_linear_second=slope_lin_smax,
            rmse_quadratic_second=rmse_quad_smax,
            r2_quadratic_second=r2_quad_smax,
            rmse_segmented_second=rmse_seg_smax,
            r2_segmented_second=r2_seg_smax,
            seg_slope_low_second=s_low_smax,
            seg_slope_high_second=s_high_smax,
            seg_threshold_second=thr_smax,
            sm_binned_min_second=sm_min_smax,
            sm_binned_max_second=sm_max_smax,
            thr_final_second=thr_final_smax,
            n_valid_second=n_valid_smax,
        ),
        coords=dict(
            lat=gpp["lat"],
            lon=gpp["lon"],
        ),
        attrs=dict(
            note="Threshold metrics for GPP-SM relationship"
        )
    )

    return ds_all


# ==========================================================
# 10. Main workflow
# ==========================================================
def main():
    """
    Main workflow:
    1. Read GPP and SM
    2. Align time
    3. Apply threshold calculation to each pixel
    4. Save all outputs into a NetCDF file
    """
    print("Loading data...")
    gpp = load_dataarray(GPP_PATH, GPP_VAR)
    sm = load_dataarray(SM_PATH, SM_VAR) * SM_SCALE_FACTOR

    # Rechunk for parallel computation
    gpp = gpp.chunk({"time": TIME_CHUNK, "lat": LAT_CHUNK, "lon": LON_CHUNK})
    sm = sm.chunk({"time": TIME_CHUNK, "lat": TIME_CHUNK if False else LAT_CHUNK, "lon": LON_CHUNK})

    print("GPP:")
    print(gpp)
    print("SM:")
    print(sm)

    # Align time coordinates
    gpp, sm = validate_and_align_time(gpp, sm)

    # Double check time equality after reassignment
    t_gpp = np.asarray(gpp.time.values)
    t_sm = np.asarray(sm.time.values)
    if not np.array_equal(t_gpp, t_sm):
        raise ValueError("Time coordinates are still not aligned after assignment.")

    print("Applying threshold computation across all grid cells...")
    out = xr.apply_ufunc(
        compute_threshold,
        gpp,
        sm,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]] * 28,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[OUTPUT_DTYPE] * 28,
        kwargs=dict(
            use_sg_filter=USE_SG_FILTER,
            min_per_bin=MIN_PER_BIN,
            min_bins=MIN_BINS,
            bin_width=BIN_WIDTH,
            sg_window=SG_WINDOW,
            sg_polyorder=SG_POLYORDER,
            debug=DEBUG,
        ),
    )

    print("Building output dataset...")
    ds_all = build_output_dataset(gpp, out)
    print(ds_all)

    print(f"Saving to: {OUTPUT_PATH}")
    encoding = {
        var: {
            "dtype": "float32",
            "_FillValue": np.nan,
            "zlib": ZLIB,
            "complevel": COMPLEVEL,
        }
        for var in ds_all.data_vars
    }

    with dask.config.set(scheduler=DASK_SCHEDULER):
        with ProgressBar():
            ds_all.to_netcdf(OUTPUT_PATH, encoding=encoding)

    print("Done.")


if __name__ == "__main__":
    main()