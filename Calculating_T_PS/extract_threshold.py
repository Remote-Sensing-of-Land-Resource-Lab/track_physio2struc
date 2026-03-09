#%%
import warnings
from numpy.polynomial.polyutils import RankWarning
from scipy.optimize import OptimizeWarning

# ---- Silence all fitting / numerical warnings ----
warnings.filterwarnings("ignore", category=RankWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def compute_threshold(gpp_ts, swc_ts, *,
                     use_sg_filter=False,
                     min_per_bin=1,
                     min_bins=6,
                     bin_width=1.0,
                     sg_window=5,
                     sg_polyorder=2,
                     debug=False):
    """
    Compute multiple GPP–SWC threshold metrics with:
      - Linear, quadratic, and segmented fits for start→max and start→end.
      - BIC for model comparison.
      - Two-break segmented model for start→end only if last 4 bins decrease.
      - Returns 34 outputs including separate final thresholds for smax and send,
        and counts of valid binned points for each.
    """

    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter

    # ==========================================================
    # --- Helper functions ---
    # ==========================================================
    def _r2_rmse(y, yhat):
        resid = y - yhat
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(ss_res / len(y))
        return r2, rmse

    def _bic(y, yhat, k):
        n = len(y)
        if n <= k + 1:
            return np.inf
        rss = np.sum((y - yhat)**2)
        return n * np.log(rss / n + 1e-15) + k * np.log(n)

    def seg1_func(x, a1, b_ratio, x0, c):
        b = a1 * b_ratio
        return np.where(x < x0, a1*x + c, b*(x - x0) + (a1*x0 + c))

    def seg2_func(x, a1, b1_ratio, x1, b2_ratio, x2, c):
        a2 = a1 * b1_ratio
        a3 = a2 * b2_ratio
        y1 = a1*x + c
        y2 = a2*(x - x1) + (a1*x1 + c)
        y3 = a3*(x - x2) + (a2*(x2 - x1) + (a1*x1 + c))
        return np.where(x < x1, y1, np.where(x < x2, y2, y3))

    def _bin_trim_with_counts(swc, gpp):
        m0 = np.isfinite(swc) & np.isfinite(gpp)
        swc0, gpp0 = swc[m0], gpp[m0]
        if swc0.size < min_per_bin * min_bins:
            return None, None, None
        lo, hi = np.floor(np.nanmin(swc0)), np.ceil(np.nanmax(swc0))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None, None, None
        edges = np.arange(lo, hi + bin_width, bin_width)
        if edges.size < 3:
            return None, None, None

        x_list, y_list, n_list = [], [], []
        bin_id = np.digitize(swc0, edges) - 1
        for bid in np.unique(bin_id):
            if bid < 0 or bid >= len(edges) - 1:
                continue
            sel = bin_id == bid
            if sel.sum() < min_per_bin:
                continue
            xx, yy = swc0[sel], gpp0[sel]
            keep = (yy <= 100) & (yy >= -100)
            if keep.sum() < min_per_bin:
                continue
            x_list.append(xx[keep].mean())
            y_list.append(yy[keep].mean())
            n_list.append(keep.sum())

        if len(x_list) < min_bins:
            return None, None, None

        x, y = np.asarray(x_list), np.asarray(y_list)
        o = np.argsort(x)
        return x[o], y[o], np.asarray(n_list)[o]

    def _fit_poly(x, y, deg, enforce_positive=False):
        try:
            p = np.polyfit(x, y, deg)
            yhat = np.polyval(p, x)
            r2, rmse = _r2_rmse(y, yhat)
            bic = _bic(y, yhat, deg + 1)
            slope = p[-2] if deg >= 1 else np.nan
            if enforce_positive and slope <= 0:
                r2, bic = -1, np.inf
            return bic, rmse, r2, slope
        except Exception:
            return np.inf, np.nan, np.nan, np.nan

    def _fit_seg1(x, y):
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
            bic = _bic(y, yhat, 4)
            return bic, rmse, r2, a1, a2, x0, sm_min, sm_max
        except Exception:
            return np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    def _fit_seg2(x, y):
        try:
            sm_min, sm_max = float(np.min(x)), float(np.max(x))
            slope_init = max((y[-1] - y[0]) / max(x[-1] - x[0], 1e-12), 1e-6)
            x1_init = x[int(len(x) * 0.4)]
            x2_init = x[int(len(x) * 0.8)]
            c_init = float(np.interp(x1_init, x, y))
            p0 = [slope_init, 0.5, x1_init, 0.5, x2_init, c_init]
            bounds = ([0, 0, sm_min, 0, sm_min, -np.inf],
                      [np.inf, 1.0, sm_max, 1.0, sm_max, np.inf])
            popt, _ = curve_fit(seg2_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
            a1, b1_ratio, x1, b2_ratio, x2, c = popt
            a2 = a1 * b1_ratio
            a3 = a2 * b2_ratio
            yhat = seg2_func(x, *popt)
            r2, rmse = _r2_rmse(y, yhat)
            bic = _bic(y, yhat, 6)
            return bic, rmse, r2, a1, a2, a3, x1, x2, sm_min, sm_max
        except Exception:
            return np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # ==========================================================
    # --- Data preparation ---
    # ==========================================================
    gpp = np.asarray(gpp_ts, float)
    swc = np.asarray(swc_ts, float)
    xb, yb, nb = _bin_trim_with_counts(swc, gpp)
    if xb is None:
        return (np.nan,) * 34

    # NEW: number of valid binned points
    n_valid_smax = len(xb[:int(np.argmax(yb)) + 1])  # start→max
    n_valid_send = len(xb)                           # start→end

    # Smooth if requested
    if use_sg_filter and len(yb) >= 5:
        win = sg_window if sg_window % 2 == 1 else sg_window + 1
        win = min(win, len(yb))
        yb_s = savgol_filter(yb, win, min(sg_polyorder, win - 1))
    else:
        yb_s = yb.copy()

    i_max = int(np.argmax(yb_s))
    xb_smax, yb_smax = xb[:i_max + 1], yb[:i_max + 1]
    xb_send, yb_send = xb, yb

    # ==========================================================
    # --- Start→Max fits ---
    # ==========================================================
    bic_lin_smax, rmse_lin_smax, r2_lin_smax, slope_lin_smax = _fit_poly(xb_smax, yb_smax, 1, enforce_positive=True)
    bic_quad_smax, rmse_quad_smax, r2_quad_smax, _ = _fit_poly(xb_smax, yb_smax, 2)
    bic_seg_smax, rmse_seg_smax, r2_seg_smax, s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax = _fit_seg1(xb_smax, yb_smax)
    thr_final_smax = (
        thr_smax if (
            np.isfinite(thr_smax)
            and np.isfinite(sm_min_smax)
            and np.isfinite(sm_max_smax)
            and (thr_smax > sm_min_smax)
            and (thr_smax < sm_max_smax)
        ) else np.nan
    )

    # ==========================================================
    # --- Start→End fits ---
    # ==========================================================
    bic_lin_send, rmse_lin_send, r2_lin_send, slope_lin_send = _fit_poly(xb_send, yb_send, 1, enforce_positive=True)
    bic_quad_send, rmse_quad_send, r2_quad_send, _ = _fit_poly(xb_send, yb_send, 2)

    # --- initialize all variables to avoid UnboundLocalError ---
    bic_seg_send = rmse_seg_send = r2_seg_send = np.nan
    s_low_send = s_high_send = thr_send = sm_min_send = sm_max_send = np.nan

    is_decreasing = np.all(np.diff(yb_send[-4:]) < 0)
    if is_decreasing:
        try:
            (bic_seg_send, rmse_seg_send, r2_seg_send,
             s1_send, s2_send, s3_send, thr1_send, thr2_send,
             sm_min_send, sm_max_send) = _fit_seg2(xb_send, yb_send)
            thr_send, s_low_send, s_high_send = thr1_send, s1_send, s2_send
        except Exception:
            pass
    else:
        try:
            (bic_seg_send, rmse_seg_send, r2_seg_send,
             s_low_send, s_high_send, thr_send,
             sm_min_send, sm_max_send) = _fit_seg1(xb_send, yb_send)
        except Exception:
            pass

    thr_final_send = (
        thr_send if (
            np.isfinite(thr_send)
            and np.isfinite(sm_min_send)
            and np.isfinite(sm_max_send)
            and (thr_send >= sm_min_send)
            and (thr_send <= sm_max_send)
        ) else np.nan
    )

    # ==========================================================
    # --- Return outputs (now 34 total) ---
    # ==========================================================
    return (
        bic_lin_smax, rmse_lin_smax, r2_lin_smax, slope_lin_smax,
        bic_quad_smax, rmse_quad_smax, r2_quad_smax,
        bic_seg_smax, rmse_seg_smax, r2_seg_smax,
        s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax, thr_final_smax,

        bic_lin_send, rmse_lin_send, r2_lin_send, slope_lin_send,
        bic_quad_send, rmse_quad_send, r2_quad_send,
        bic_seg_send, rmse_seg_send, r2_seg_send,
        s_low_send, s_high_send, thr_send, sm_min_send, sm_max_send, thr_final_send,

        n_valid_smax, n_valid_send
    )

#%%
import os
import numpy as np
import xarray as xr

GPP_PATH = "gs_drought_gpp_zscore.nc"
SWC_PATH = "gs_drought_swc_8days_1982_2013.nc"
GPP_VAR  = "GPP"
SWC_VAR  = "swc"

gpp = xr.open_dataset(GPP_PATH, chunks={})[GPP_VAR]
swc = xr.open_dataset(SWC_PATH, chunks={})[SWC_VAR]*100

gpp = gpp.chunk({"time": -1, "lat": 128, "lon": 128})
swc = swc.chunk({"time": -1, "lat": 128, "lon": 128})

print(gpp, swc)

swc = swc.assign_coords(time = gpp.time)

#%%
import pandas as pd
df = pd.DataFrame({'ndvi': gpp.time.values, 'swc': swc.time.values})
print(df[df['ndvi'] != df['swc']])

#%%
t_nirv = pd.to_datetime(gpp.time.values)
t_swc  = pd.to_datetime(swc.time.values)
missing_in_nirv = np.setdiff1d(t_swc, t_nirv)
missing_in_swc  = np.setdiff1d(t_nirv, t_swc)
print(f"🟠 Dates missing in NIRv (present in SWC only): {missing_in_nirv}")
print(f"🔵 Dates missing in SWC (present in NIRv only): {missing_in_swc}")

#%%
# --- Apply the function across all grid cells ---
out = xr.apply_ufunc(
    compute_threshold,
    gpp, swc,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[]] * 34,
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float32] * 34,
)

(
    bic_lin_smax, rmse_lin_smax, r2_lin_smax, slope_lin_smax,
    bic_quad_smax, rmse_quad_smax, r2_quad_smax,
    bic_seg_smax, rmse_seg_smax, r2_seg_smax,
    s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax, thr_final_smax,

    bic_lin_send, rmse_lin_send, r2_lin_send, slope_lin_send,
    bic_quad_send, rmse_quad_send, r2_quad_send,
    bic_seg_send, rmse_seg_send, r2_seg_send,
    s_low_send, s_high_send, thr_send, sm_min_send, sm_max_send, thr_final_send,

    n_valid_smax, n_valid_send
) = out

# ==========================================================
# --- Combine results ---
# ==========================================================
ds_all = xr.Dataset(
    data_vars=dict(
        # Primary (start→end)
        bic_linear=bic_lin_send,
        rmse_linear=rmse_lin_send,
        r2_linear=r2_lin_send,
        slope_linear=slope_lin_send,
        bic_quadratic=bic_quad_send,
        rmse_quadratic=bic_quad_send,
        r2_quadratic=r2_quad_send,
        bic_segmented=bic_seg_send,
        rmse_segmented=rmse_seg_send,
        r2_segmented=r2_seg_send,
        seg_slope_low=s_low_send,
        seg_slope_high=s_high_send,
        seg_threshold=thr_send,
        sm_binned_min=sm_min_send,
        sm_binned_max=sm_max_send,
        thr_final=thr_final_send,
        n_valid=n_valid_send,

        # Secondary (start→max)
        bic_linear_second=bic_lin_smax,
        rmse_linear_second=rmse_lin_smax,
        r2_linear_second=r2_lin_smax,
        slope_linear_second=slope_lin_smax,
        bic_quadratic_second=bic_quad_smax,
        rmse_quadratic_second=rmse_quad_smax,
        r2_quadratic_second=r2_quad_smax,
        bic_segmented_second=bic_seg_smax,
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
    coords=dict(lat=gpp["lat"], lon=gpp["lon"]),
    attrs=dict(note="Threshold metrics for GPP–SWC relationship; includes valid binned counts")
)

print(ds_all)

#%%
import dask
from dask.diagnostics import ProgressBar

enc = {v: {"dtype": "float32", "_FillValue": np.nan, "zlib": True, "complevel": 4}
       for v in ds_all.data_vars}

with dask.config.set(scheduler="threads"):
    with ProgressBar():
        ds_all.to_netcdf("threshold_metrics_zscore.nc", encoding=enc)
