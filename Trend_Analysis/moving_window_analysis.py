# ==========================================================
# MOVING-WINDOW GPP–SWC THRESHOLD ANALYSIS (FINAL VERSION)
# Output: All 34 metrics for every pixel and every window
# ==========================================================

import numpy as np
import xarray as xr
import warnings
from scipy.optimize import OptimizeWarning
from numpy.polynomial.polyutils import RankWarning

# Silence warnings for stability
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RankWarning)


# ==========================================================
# 1. Enhanced compute_threshold (34 outputs)
# ==========================================================

def compute_threshold(gpp_ts, swc_ts, *,
                     min_per_bin=1,
                     min_bins=6,
                     bin_width=1.0,
                     use_sg_filter=False,
                     sg_window=5,
                     sg_polyorder=2):
    """
    Compute 34 GPP–SWC relationship metrics:
    - linear, quadratic, segmented fits (start→max)
    - linear, quadratic, segmented fits (start→end)
    - two-break segmented model if the right tail decreases
    - Boundary checking for thresholds
    - n_valid_smax, n_valid_send
    """

    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit

    # ======================================================
    # Helper functions
    # ======================================================
    def _r2_rmse(y, yhat):
        resid = y - yhat
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(ss_res / len(y))
        return r2, rmse

    def _bic(y, yhat, k):
        n = len(y)
        rss = np.sum((y - yhat)**2)
        return n * np.log(rss / n + 1e-15) + k * np.log(n)

    def seg1_func(x, a1, b_ratio, x0, c):
        b = a1 * b_ratio
        return np.where(x < x0, a1*x + c,
                        b*(x-x0) + (a1*x0 + c))

    def seg2_func(x, a1, b1_ratio, x1, b2_ratio, x2, c):
        a2 = a1 * b1_ratio
        a3 = a2 * b2_ratio
        y1 = a1 * x + c
        y2 = a2 * (x - x1) + (a1 * x1 + c)
        y3 = a3 * (x - x2) + (a2*(x2-x1) + (a1*x1 + c))
        return np.where(x < x1, y1,
                        np.where(x < x2, y2, y3))

    # ======================================================
    # Bin mask
    # ======================================================
    def _bin_trim(swc, gpp):
        m0 = np.isfinite(swc) & np.isfinite(gpp)
        swc0, gpp0 = swc[m0], gpp[m0]
        if swc0.size < min_bins * min_per_bin:
            return None, None, None
        lo = np.floor(np.nanmin(swc0))
        hi = np.ceil(np.nanmax(swc0))
        if lo == hi:
            return None, None, None
        edges = np.arange(lo, hi + bin_width, bin_width)
        x_list, y_list, n_list = [], [], []

        bin_id = np.digitize(swc0, edges) - 1
        for bid in np.unique(bin_id):
            if bid < 0 or bid >= len(edges)-1:
                continue
            sel = bin_id == bid
            if sel.sum() < min_per_bin:
                continue
            xx, yy = swc0[sel], gpp0[sel]
            keep = (yy < 100) & (yy > -100)
            if keep.sum() < min_per_bin:
                continue
            x_list.append(xx[keep].mean())
            y_list.append(yy[keep].mean())
            n_list.append(keep.sum())

        if len(x_list) < min_bins:
            return None, None, None

        x, y = np.array(x_list), np.array(y_list)
        o = np.argsort(x)
        return x[o], y[o], np.array(n_list)[o]

    # ======================================================
    # Main part
    # ======================================================
    gpp = np.asarray(gpp_ts, float)
    swc = np.asarray(swc_ts, float)

    xb, yb, nb = _bin_trim(swc, gpp)
    if xb is None:
        return (np.nan,) * 34

    # smooth?
    if use_sg_filter and len(yb) >= 5:
        win = min(sg_window if sg_window % 2 == 1 else sg_window+1, len(yb))
        yb_s = savgol_filter(yb, win, min(sg_polyorder, win-1))
    else:
        yb_s = yb.copy()

    i_max = int(np.argmax(yb_s))
    xb_smax, yb_smax = xb[:i_max+1], yb[:i_max+1]
    xb_send, yb_send = xb, yb

    n_valid_smax = len(xb_smax)
    n_valid_send = len(xb_send)

    # ======================================================
    # Fit helpers
    # ======================================================
    def _fit_poly(x, y, deg, enforce_positive=False):
        try:
            p = np.polyfit(x, y, deg)
            yhat = np.polyval(p, x)
            r2, rmse = _r2_rmse(y, yhat)
            bic = _bic(y, yhat, deg+1)
            slope = p[-2] if deg >= 1 else np.nan
            if enforce_positive and slope <= 0:
                return np.inf, np.nan, np.nan, np.nan
            return bic, rmse, r2, slope
        except:
            return np.inf, np.nan, np.nan, np.nan

    def _fit_seg1(x, y):
        try:
            sm_min, sm_max = float(x.min()), float(x.max())
            slope0 = max((y[-1]-y[0]) / max(x[-1]-x[0], 1e-12), 1e-6)
            x0_init = float(np.median(x))
            c_init = float(np.interp(x0_init, x, y))
            p0 = [slope0, 0.5, x0_init, c_init]
            bounds = ([0,0,sm_min,-np.inf],[np.inf,1,sm_max,np.inf])
            popt,_ = curve_fit(seg1_func, x, y, p0=p0, bounds=bounds, maxfev=2000)
            a1,b_ratio,x0,c = popt
            a2 = a1*b_ratio
            yhat = seg1_func(x,*popt)
            r2,rmse = _r2_rmse(y,yhat)
            bic = _bic(y,yhat,4)
            return bic,rmse,r2,a1,a2,x0,sm_min,sm_max
        except:
            return np.inf,*([np.nan]*7)

    def _fit_seg2(x, y):
        try:
            sm_min, sm_max = float(x.min()), float(x.max())
            slope0 = max((y[-1]-y[0])/max(x[-1]-x[0],1e-12),1e-6)
            x1_init = x[int(len(x)*0.4)]
            x2_init = x[int(len(x)*0.8)]
            c_init = float(np.interp(x1_init, x, y))
            p0 = [slope0,0.5,x1_init,0.5,x2_init,c_init]
            bounds=([0,0,sm_min,0,sm_min,-np.inf],[np.inf,1,sm_max,1,sm_max,np.inf])
            popt,_ = curve_fit(seg2_func,x,y,p0=p0,bounds=bounds,maxfev=5000)
            a1,b1r,x1,b2r,x2,c = popt
            a2 = a1*b1r
            a3 = a2*b2r
            yhat = seg2_func(x,*popt)
            r2,rmse = _r2_rmse(y,yhat)
            bic = _bic(y,yhat,6)
            return bic,rmse,r2,a1,a2,a3,x1,x2,sm_min,sm_max
        except:
            return np.inf,*([np.nan]*9)

    # ======================================================
    # Start→Max fits
    # ======================================================
    bic_lin_smax, rmse_lin_smax, r2_lin_smax, slope_lin_smax = _fit_poly(xb_smax, yb_smax, 1)
    bic_quad_smax, rmse_quad_smax, r2_quad_smax, _ = _fit_poly(xb_smax, yb_smax, 2)
    (bic_seg_smax, rmse_seg_smax, r2_seg_smax,
     s_low_smax, s_high_smax, thr_smax, sm_min_smax, sm_max_smax) = _fit_seg1(xb_smax, yb_smax)

    thr_final_smax = thr_smax if (np.isfinite(thr_smax)
                                and sm_min_smax < thr_smax < sm_max_smax) else np.nan

    # ======================================================
    # Start→End fits
    # ======================================================
    bic_lin_send, rmse_lin_send, r2_lin_send, slope_lin_send = _fit_poly(xb_send, yb_send, 1)
    bic_quad_send, rmse_quad_send, r2_quad_send, _ = _fit_poly(xb_send, yb_send, 2)

    bic_seg_send = rmse_seg_send = r2_seg_send = np.nan
    s_low_send = s_high_send = thr_send = np.nan
    sm_min_send = sm_max_send = np.nan

    is_decreasing = np.all(np.diff(yb_send[-4:]) < 0)
    if is_decreasing:
        (bic_seg_send, rmse_seg_send, r2_seg_send,
         s1, s2, s3, thr1, thr2, sm_min_send, sm_max_send) = _fit_seg2(xb_send, yb_send)
        thr_send, s_low_send, s_high_send = thr1, s1, s2
    else:
        (bic_seg_send, rmse_seg_send, r2_seg_send,
         s_low_send, s_high_send, thr_send,
         sm_min_send, sm_max_send) = _fit_seg1(xb_send, yb_send)

    thr_final_send = thr_send if (np.isfinite(thr_send)
                                  and sm_min_send < thr_send < sm_max_send) else np.nan

    # ======================================================
    # Return 34 metrics
    # ======================================================
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


# ==========================================================
# 2. Moving window wrapper
# ==========================================================

def compute_threshold_mw(gpp_ts, swc_ts, time, window=15):
    years = time.astype("datetime64[Y]").astype(int)
    unique_years = np.unique(years)
    n_windows = len(unique_years) - window + 1
    if n_windows <= 0:
        return np.full((1, 34), np.nan)

    results = []
    for i in range(n_windows):
        start_y = unique_years[i]
        end_y = unique_years[i+window-1]
        mask = (years >= start_y) & (years <= end_y)
        res = compute_threshold(gpp_ts[mask], swc_ts[mask])
        results.append(res)
    return np.array(results)


# ==========================================================
# 3. Full-region analysis
# ==========================================================

def run_threshold_analysis(GPP_PATH, SWC_PATH,
                           GPP_VAR="GPP",
                           SWC_VAR="swc",
                           window=15,
                           out_fname="threshold_mw.nc"):

    print("🌱 Loading data...")
    gpp = xr.open_dataset(GPP_PATH)[GPP_VAR]
    swc = xr.open_dataset(SWC_PATH)[SWC_VAR]*100

    # scale SWC if needed (user adjust)
    # swc = swc * 100

    gpp = gpp.chunk({"time": -1, "lat": 128, "lon": 128})
    swc = swc.chunk({"time": -1, "lat": 128, "lon": 128})

    time = gpp.time.values

    print("🚀 Running moving-window threshold extraction...")

    n_metrics = 34

    # Extract unique years
    years = np.unique(gpp["time.year"].values)
    n_years = len(years)
    n_windows = n_years - window + 1

    print("n_windows =", n_windows)

    from functools import partial
    mw_func = partial(compute_threshold_mw, window=window)

    out = xr.apply_ufunc(
        mw_func,
        gpp, swc, gpp["time"],
        input_core_dims=[["time"], ["time"], ["time"]],
        output_core_dims=[["window", "metric"]],
        output_sizes={"window": n_windows, "metric": n_metrics},  # ★ FIX ★
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    print(out)
    # window centers
    years = np.unique(gpp["time.year"].values)
    centers = np.array([
        int((years[i] + years[i + window - 1]) / 2)
        for i in range(len(years) - window + 1)
    ])

    metric_names = [
        "bic_lin_smax","rmse_lin_smax","r2_lin_smax","slope_lin_smax",
        "bic_quad_smax","rmse_quad_smax","r2_quad_smax",
        "bic_seg_smax","rmse_seg_smax","r2_seg_smax",
        "s_low_smax","s_high_smax","thr_smax","sm_min_smax","sm_max_smax","thr_final_smax",

        "bic_lin_send","rmse_lin_send","r2_lin_send","slope_lin_send",
        "bic_quad_send","rmse_quad_send","r2_quad_send",
        "bic_seg_send","rmse_seg_send","r2_seg_send",
        "s_low_send","s_high_send","thr_send","sm_min_send","sm_max_send","thr_final_send",

        "n_valid_smax","n_valid_send"
    ]

    ds = xr.Dataset(
        {name: out.isel(metric=i).assign_coords(window=("window", centers))
         for i, name in enumerate(metric_names)},
        coords=dict(lat=gpp.lat, lon=gpp.lon, window=("window", centers)),
        attrs=dict(info="Moving-window GPP–SWC threshold metrics")
    )

    print("💾 Saving:", out_fname)
    enc = {v: {"zlib": True, "complevel":4, "_FillValue": np.nan}
           for v in ds.data_vars}

    from dask.diagnostics import ProgressBar
    with ProgressBar():
        ds.to_netcdf(out_fname, encoding=enc)

    print("🎉 DONE:", out_fname)
    return ds


# ==========================================================
# 4. RUN
# ==========================================================

ds = run_threshold_analysis(
    GPP_PATH="gs_drought_gpp_mapped_1982_2022.nc",
    SWC_PATH="gs_drought_swc_1982_2022.nc",
    GPP_VAR="GPP",
    SWC_VAR="swc",
    window=15,
    out_fname="threshold_metrics_8day_trend_gosif_10_1.nc"
)

print(ds)
