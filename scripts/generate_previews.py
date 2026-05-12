"""
generate_previews.py
====================
Generates publication-quality preview images from the band-1 and band-3
orthophotos already available NOW (before bands 2,4,5 finish).

Creates:
  1. Grid overview   — all 10 plots side by side (band 3 = Red channel)
  2. Per-plot panels — true-color proxy + NDVI-proxy for B01 & B05
  3. Pipeline summary poster — suitable for showing professors

Run now (only needs bands 1 and 3):
    pip install rasterio numpy matplotlib pillow --quiet
    python3 ~/Desktop/detectree2/scripts/generate_previews.py

Outputs go to:
    ~/Desktop/detectree2/previews/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings("ignore")

ODM_DIR     = Path("/home/cdac/Desktop/detectree2/data/odm_projects")
PREVIEW_DIR = Path("/home/cdac/Desktop/detectree2/previews")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

PLOTS = [f"plot_B{i:02d}" for i in range(1, 11)]

SPLIT_COLOR = {
    "train": "#3fb950",   # green
    "val":   "#f0883e",   # orange
    "test":  "#58a6ff",   # blue
}
SPLIT_LABEL = {
    "plot_B01": "train", "plot_B02": "train", "plot_B03": "train",
    "plot_B04": "train", "plot_B05": "train",
    "plot_B06": "val",   "plot_B07": "val",
    "plot_B08": "test",  "plot_B09": "test",  "plot_B10": "test",
}

BG   = "#0d1117"
TEXT = "#e6edf3"
ACC  = "#3fb950"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": "#30363d",
})

# ── helpers ──────────────────────────────────────────────────────────────────

def load_thumb(plot, band, size=512):
    """Load a single-band orthophoto, return normalised uint8 thumbnail."""
    p = ODM_DIR / f"{plot}_band{band}" / "odm_orthophoto" / "odm_orthophoto.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        # Compute downsampled read shape
        scale = size / max(src.width, src.height)
        out_w = max(1, int(src.width  * scale))
        out_h = max(1, int(src.height * scale))
        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average
        ).astype(np.float32)
    # Stretch
    lo, hi = np.nanpercentile(arr[arr > 0], [2, 98]) if arr[arr > 0].size else (0, 1)
    arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return arr

def make_rgb_proxy(plot, size=512):
    """Make an RGB proxy image from bands 3(R), 2(G), 1(B) if available, else greyscale."""
    b3 = load_thumb(plot, 3, size)
    b2 = load_thumb(plot, 2, size)
    b1 = load_thumb(plot, 1, size)

    if b3 is None:
        return None

    if b2 is not None and b1 is not None:
        # True-color proxy: stack R,G,B
        h = min(b3.shape[0], b2.shape[0], b1.shape[0])
        w = min(b3.shape[1], b2.shape[1], b1.shape[1])
        rgb = np.stack([b3[:h,:w], b2[:h,:w], b1[:h,:w]], axis=-1)
    else:
        # Greyscale from band3 only
        rgb = np.stack([b3, b3, b3], axis=-1)
    return rgb

def ndvi_proxy(plot, size=512):
    """Compute NDVI proxy from band5(NIR) and band3(Red)."""
    nir = load_thumb(plot, 5, size)
    red = load_thumb(plot, 3, size)
    if nir is None or red is None:
        return None
    h = min(nir.shape[0], red.shape[0])
    w = min(nir.shape[1], red.shape[1])
    num = nir[:h,:w] - red[:h,:w]
    den = nir[:h,:w] + red[:h,:w] + 1e-9
    return np.clip(num / den, -1, 1)


# ── Figure 1: 10-plot overview grid ─────────────────────────────────────────

def fig_overview():
    print("  Generating overview grid...")
    fig = plt.figure(figsize=(22, 10), facecolor=BG)
    fig.suptitle(
        "UAV Multispectral Survey — Cocoa Agroforestry Plots (Divo, Côte d'Ivoire)\n"
        "Band 3 · Red channel · 5 cm GSD · DJI Phantom 4 Multispectral",
        fontsize=14, color=TEXT, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.12, wspace=0.05,
                           left=0.02, right=0.98, top=0.88, bottom=0.08)

    for i, plot in enumerate(PLOTS):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        thumb = load_thumb(plot, 3, size=400)
        split = SPLIT_LABEL.get(plot, "train")
        col   = SPLIT_COLOR[split]

        if thumb is not None:
            ax.imshow(thumb, cmap="gray", interpolation="bilinear")
        else:
            ax.set_facecolor("#161b22")
            ax.text(0.5, 0.5, "processing…", ha="center", va="center",
                    color="#8b949e", fontsize=9)

        # Coloured border indicating split
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2.5)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{plot.replace('_', ' ')}",
                     color=col, fontsize=10, fontweight="bold", pad=3)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=SPLIT_COLOR[s], label=s.upper())
                  for s in ["train", "val", "test"]]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor=TEXT, fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    out = PREVIEW_DIR / "01_overview_all_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {out.name}")
    return out


# ── Figure 2: Single-plot detail panel ───────────────────────────────────────

def fig_plot_detail(plot_name):
    print(f"  Generating detail panel for {plot_name}...")
    rgb  = make_rgb_proxy(plot_name, size=600)
    ndvi = ndvi_proxy(plot_name, size=600)
    b1   = load_thumb(plot_name, 1, size=600)

    n_panels = sum(x is not None for x in [rgb, ndvi, b1])
    if n_panels == 0:
        print(f"  ⚠  No data for {plot_name}")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    split = SPLIT_LABEL.get(plot_name, "train")
    col   = SPLIT_COLOR[split]

    fig.suptitle(
        f"{plot_name.replace('_', ' ')} · Split: {split.upper()}",
        fontsize=15, color=col, fontweight="bold"
    )

    # Panel 1: RGB or greyscale
    axes[0].imshow(rgb if rgb is not None else np.zeros((100,100,3)),
                   interpolation="bilinear")
    axes[0].set_title("True-Color Proxy (R-G-B)", color=ACC, fontsize=12)
    axes[0].axis("off")

    # Panel 2: Band 1 (Blue)
    if b1 is not None:
        axes[1].imshow(b1, cmap="Blues_r", interpolation="bilinear")
    else:
        axes[1].set_facecolor("#161b22")
        axes[1].text(0.5, 0.5, "Band 1\n(processing…)",
                     ha="center", va="center", color="#8b949e")
    axes[1].set_title("Band 1 · Blue 450 nm", color=ACC, fontsize=12)
    axes[1].axis("off")

    # Panel 3: NDVI proxy
    if ndvi is not None:
        im = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-0.3, vmax=0.8,
                            interpolation="bilinear")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                     label="NDVI proxy")
        axes[2].set_title("NDVI Proxy (NIR−R)/(NIR+R)", color=ACC, fontsize=12)
    else:
        axes[2].set_facecolor("#161b22")
        axes[2].text(0.5, 0.5, "NDVI\n(needs Band 5)", ha="center",
                     va="center", color="#8b949e", fontsize=11)
        axes[2].set_title("NDVI Proxy", color=ACC, fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out = PREVIEW_DIR / f"02_detail_{plot_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {out.name}")
    return out


# ── Figure 3: Pipeline summary poster ────────────────────────────────────────

def fig_pipeline_poster():
    print("  Generating pipeline summary poster...")
    fig = plt.figure(figsize=(20, 11), facecolor=BG)

    # ── Title ──
    fig.text(0.5, 0.96,
             "Individual Tree Detection in Cocoa Agroforestry",
             ha="center", fontsize=20, color=TEXT,
             fontweight="bold")
    fig.text(0.5, 0.92,
             "detectree2 · Mask R-CNN · DJI Phantom 4 Multispectral · 5 cm GSD",
             ha="center", fontsize=12, color="#8b949e")

    # ── Pipeline flow (text boxes) ──
    steps = [
        ("1. Raw Images\n(JPG + 5-band TIFF)", "#1f6feb"),
        ("2. ODM\nOrthomosaics\n(5 cm/px)", "#388bfd"),
        ("3. Band Stacking\n5-band GeoTIFF\nper plot", "#58a6ff"),
        ("4. Tiling\n512×512 px\noverlapping tiles", "#79c0ff"),
        ("5. Mask R-CNN\n(detectree2)\nFine-tuning", ACC),
        ("6. Predictions\nTree crowns\n+ polygons", "#56d364"),
    ]

    xs = np.linspace(0.07, 0.93, len(steps))
    y_box = 0.60
    box_w, box_h = 0.11, 0.20

    for i, (label, color) in enumerate(steps):
        x = xs[i]
        fancy = FancyBboxPatch(
            (x - box_w/2, y_box - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.01", linewidth=2,
            edgecolor=color, facecolor=color + "22",
            transform=fig.transFigure, clip_on=False
        )
        fig.add_artist(fancy)
        fig.text(x, y_box, label, ha="center", va="center",
                 fontsize=9.5, color=color, fontweight="bold",
                 multialignment="center")
        if i < len(steps) - 1:
            fig.annotate("",
                xy=(xs[i+1] - box_w/2 - 0.005, y_box),
                xytext=(x + box_w/2 + 0.005, y_box),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=2),
            )

    # ── Dataset stats table ──
    stats = [
        ["Metric", "Value"],
        ["Plots",                "10 (B01–B10)"],
        ["Total RGB images",     "1,373"],
        ["Total Multispectral",  "6,365 (5 bands × 1,273)"],
        ["Area covered",         "~30 ha"],
        ["GSD",                  "4.2–4.6 cm"],
        ["Flight altitude",      "80 m AGL"],
        ["Train / Val / Test",   "B01-B05 / B06-B07 / B08-B10"],
        ["ODM status",           "Bands 1 & 3 ✓  |  Bands 2,4,5 ⏳"],
    ]

    col_x = [0.10, 0.40]
    y0 = 0.44
    row_h = 0.038

    for r, row in enumerate(stats):
        y = y0 - r * row_h
        is_header = (r == 0)
        for c, cell in enumerate(row):
            fig.text(col_x[c], y, cell,
                     ha="left", va="center",
                     fontsize=10 if not is_header else 11,
                     color=ACC if is_header else TEXT,
                     fontweight="bold" if is_header else "normal")
        # Row divider
        if not is_header:
            fig.add_artist(plt.Line2D(
                [col_x[0]-0.01, col_x[-1]+0.35], [y - row_h/2]*2,
                transform=fig.transFigure, color="#21262d", lw=0.8
            ))

    # ── Thumbnail strip (band 3 of B01, B03, B05, B07, B09) ──
    thumb_plots = ["plot_B01", "plot_B03", "plot_B05", "plot_B07", "plot_B09"]
    thumb_xs = np.linspace(0.60, 0.97, len(thumb_plots))
    thumb_y  = 0.22
    thumb_sz = 0.12

    fig.text(0.785, 0.36, "Sample orthomosaics (Band 3 · Red)",
             ha="center", fontsize=10, color="#8b949e")

    for tx, pname in zip(thumb_xs, thumb_plots):
        thumb = load_thumb(pname, 3, size=256)
        split = SPLIT_LABEL.get(pname, "train")
        col   = SPLIT_COLOR[split]

        ax = fig.add_axes([tx - thumb_sz/2, thumb_y - thumb_sz/2,
                           thumb_sz, thumb_sz])
        ax.set_facecolor("#161b22")
        if thumb is not None:
            ax.imshow(thumb, cmap="gray", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(2)
        ax.set_title(pname.replace("_"," "), fontsize=8,
                     color=col, pad=2)

    # ── Footer ──
    fig.text(0.5, 0.02,
             "Lammoglia et al. (2024) · Data in Brief 55 · DOI:10.18167/DVN1/MK2ZRG  |  "
             "detectree2 · Ball et al.  |  ODM photogrammetry",
             ha="center", fontsize=8, color="#484f58")

    out = PREVIEW_DIR / "03_pipeline_poster.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {out.name}")
    return out


# ── Figure 4: Band comparison (band 1 vs band 3 for B05) ─────────────────────

def fig_band_comparison():
    print("  Generating band comparison figure...")
    plot = "plot_B05"   # largest plot, most visually interesting
    b1 = load_thumb(plot, 1, size=500)
    b3 = load_thumb(plot, 3, size=500)

    if b1 is None or b3 is None:
        print("  ⚠  B05 band 1 or 3 missing — skipping band comparison")
        return None

    h = min(b1.shape[0], b3.shape[0])
    w = min(b1.shape[1], b3.shape[1])
    diff = b3[:h,:w] - b1[:h,:w]   # Red − Blue highlights vegetation

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(
        "plot_B05 · Band Comparison — Blue (450 nm) vs Red (650 nm)",
        fontsize=14, color=TEXT, fontweight="bold"
    )

    panels = [
        (b1[:h,:w], "Blues_r", "Band 1 · Blue  450 nm\n(Leaf reflectance / scattering)"),
        (b3[:h,:w], "Reds_r",  "Band 3 · Red   650 nm\n(Chlorophyll absorption)"),
        (diff,      "RdBu_r",  "Red − Blue difference\n(Vegetation contrast proxy)"),
    ]

    for ax, (arr, cmap, title) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, interpolation="bilinear")
        ax.set_title(title, color=ACC, fontsize=11, pad=6)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = PREVIEW_DIR / "04_band_comparison_B05.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {out.name}")
    return out


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("  GENERATING PREVIEW IMAGES")
print("  (Uses only Band 1 + Band 3 — available now)")
print("=" * 60)

fig_overview()
fig_plot_detail("plot_B01")
fig_plot_detail("plot_B05")
fig_pipeline_poster()
fig_band_comparison()

print(f"\n  All previews saved to: {PREVIEW_DIR}")
print("  Files:")
for f in sorted(PREVIEW_DIR.glob("*.png")):
    print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")
