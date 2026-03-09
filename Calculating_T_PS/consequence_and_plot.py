import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FixedLocator
import cmocean                 # pip install cmocean
import cmcrameri.cm as cmc     # pip install cmcrameri
import xarray as xr

plt.style.use("default")
# ====== Style ======
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold"
})

# ====== Load datasets ======
ds = xr.open_dataset("threshold_metrics_8daymin_full.nc")
gpp_clim = xr.open_dataset("gpp_mean.nc")['GPP'].squeeze()

# condition
cond = (ds["r2_quadratic_second"] > ds["r2_linear_second"]) & (ds['n_valid_second'] >= 6)

# slopes
r1 = ds['seg_slope_low_second']
r2 = ds['seg_slope_high_second']

# slope difference (per 0.01 SWC, already *100)
r = (-r2 + r1) * 100
r = r.where(cond)

# additional mask: thr_final_second must be non-null
valid_mask = np.isfinite(ds["thr_final_second"])
r = r.where(valid_mask)

# ====== ΔGPP loss weighted by climatology ======
# r is already per 0.01 SWC (%). Convert %→fraction then × GPP_clim (8-day)
r_weighted = (r / 100) * gpp_clim   # unit: gC m-2 8-day-1
r_weighted = r_weighted.where(valid_mask)

# ====== Value limits (your quantile-based) ======
vmin_rw = 0
vmax_rw = 5
# ============================================================
# === 1. 定义 40 个 colormap ==================================
# ============================================================

# 10 basic matplotlib colormaps
basic_base = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "YlOrBr", "OrRd", "PuRd", "GnBu", "BuPu"
]
basic_cmaps = basic_base + [c + "_r" for c in basic_base]


# 10 seaborn palettes (converted to colormap)
sns_base = [
    "rocket", "mako", "flare", "crest",
    "icefire", "vlag", "cubehelix"
]
# seaborn 有的调色板已经定义 r，但我们仍统一处理
sns_cmaps = sns_base + [c + "_r" for c in sns_base]


# 10 cmocean colormaps
cmo_base = [
    "thermal", "haline", "solar", "ice",
    "gray", "turbid", "matter", "algae",
    "balance", "curl"
]
cmo_cmaps = cmo_base + [c + "_r" for c in cmo_base]


# 10 cmcrameri scientific colormaps
cmcrameri_base = [
    "batlow", "batlowK", "devon", "oleron",
    "roma", "romaO", "vik", "broc",
    "bukavu", "tofino"
]
cmcrameri_cmaps = cmcrameri_base + [c + "_r" for c in cmcrameri_base]


# 最终合并全部
all_cmaps = basic_cmaps + sns_cmaps + cmo_cmaps + cmcrameri_cmaps

# ============================================================
# === 2. 循环绘图 =============================================
# ============================================================

for cmap_name in all_cmaps:
    print(f"Drawing with colormap: {cmap_name}")

    # ====== 选择 colormap 对象 ======
    if cmap_name in sns_cmaps:
        cmap = sns.color_palette(cmap_name, as_cmap=True)

    elif cmap_name in cmo_cmaps:
        cmap = getattr(cmocean.cm, cmap_name)

    elif cmap_name in cmcrameri_cmaps:
        cmap = getattr(cmc, cmap_name)

    else:
        cmap = plt.get_cmap(cmap_name)

    # ====== Figure ======
    fig = plt.figure(figsize=(5, 2.7))
    proj = ccrs.Robinson(central_longitude=0)
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, -60, 90])

    # ====== Base map ======
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="black")

    # ====== Gridlines ======
    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="gray", alpha=0.2, linestyle="--")
    gl.xlocator = FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = FixedLocator([-60, -30, 0, 30, 60])

    # Latitude labels
    for lat in [-30, 0, 30, 60]:
        label = f"{abs(lat)}°{'N' if lat > 0 else ('S' if lat < 0 else '')}"
        x_disp, y_disp = ax.projection.transform_point(180, lat, src_crs=ccrs.PlateCarree())
        x_disp_out = x_disp + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x_disp_out, y_disp, label, transform=ax.transData,
                fontsize=7, ha="left", va="center")

    # Longitude labels
    for lon in np.arange(-180, 181, 60):
        if lon == 0:
            label = "0°"
        elif lon < 0:
            label = f"{abs(lon)}°W"
        else:
            label = f"{lon}°E"
        ax.text(lon, -63, label, transform=ccrs.PlateCarree(),
                fontsize=7, ha="center", va="top")

    # ====== Plot ======
    im = ax.pcolormesh(
        r_weighted["lon"], r_weighted["lat"], r_weighted,
        cmap=cmap, vmin=vmin_rw, vmax=vmax_rw,
        transform=ccrs.PlateCarree(), shading="auto"
    )

    # ====== Colorbar ======
    cb = plt.colorbar(
        im, orientation="horizontal",
        fraction=0.045, pad=0.15,
        aspect=30, extend="both"
    )
    cb.ax.tick_params(length=2, labelsize=7)
    cb.set_label(
        "Difference in GPP loss (g C m$^{-2}$ day$^{-1}$) per 0.01 m$^3$/m$^{3}$ SM\n",
        fontsize=7, labelpad=5.5
    )

    # ====== Save ======
    outname = f"Panel_B_delta_GPP_loss_{cmap_name}.pdf"
    plt.tight_layout()
    plt.savefig(outname, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {outname}")
