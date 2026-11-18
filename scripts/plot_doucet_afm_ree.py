"""Reproduce the AFM and REE figure methods using the frozen D36 data."""

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/frozen/experiment_v5_clean_identity_expansion13_2026-10-26"
DATA = FROZEN / "D36_v9_clean_expansion13.parquet"
OUTPUT = ROOT / "figures/doucet_afm_ree_d36_2026-11-16"
ID = "Physical_sample_ID"
LABEL = "Secondary_label"
FE_FACTOR = 0.8998
PM = pd.Series({
    "La": 0.648, "Ce": 1.675, "Pr": 0.254, "Nd": 1.250, "Sm": 0.406,
    "Eu": 0.154, "Gd": 0.544, "Tb": 0.099, "Dy": 0.674, "Ho": 0.149,
    "Er": 0.438, "Tm": 0.068, "Yb": 0.441, "Lu": 0.0675,
}, name="PM_ppm")
GROUPS = {
    "MOR": ("Mid-ocean ridge", "#17638E"),
    "BAB": ("Back-arc basin", "#80549B"),
    "CR": ("Continental rift", "#8A5F2D"),
    "CA": ("Continental arc", "#156946"),
    "IA": ("Island arc", "#4F7836"),
    "IOA": ("Intra-oceanic arc", "#147885"),
    "CFB": ("Continental flood basalt", "#AB402E"),
    "OI": ("Ocean island", "#A43D74"),
    "OP": ("Oceanic plateau", "#4E577E"),
}
# Rickwood (1989), Table 3, digitization of Irvine & Baragar (1971) Fig. 2A.
# The tabulated coordinates are closed percentages, with all iron as FeO.
BOUNDARY = np.array([
    (58.8, 36.2, 5), (47.6, 42.4, 10), (29.6, 52.6, 17.8),
    (25.4, 54.6, 20), (21.4, 54.6, 24), (19.4, 52.8, 27.8),
    (18.9, 51.1, 30), (16.6, 43.4, 40), (15, 35, 50),
])
QUANTILES = {0: "min", 0.05: "p05", 0.25: "q1", 0.5: "median",
             0.75: "q3", 0.95: "p95", 1: "max"}
HEIGHT = np.sqrt(3) / 2


def close_afm(oxides):
    values = np.asarray(oxides, dtype=float).copy()
    values[:, 1] /= FE_FACTOR
    return values / values.sum(axis=1, keepdims=True)


def xy(values):
    _, iron, magnesia = np.asarray(values).T
    return magnesia + iron / 2, HEIGHT * iron


def load_data(path):
    ree_columns = {f"{el.upper()}(PPM)": el for el in PM.index}
    afm_columns = ["NA2O(WT%)", "K2O(WT%)", "FEOT(WT%)", "MGO(WT%)"]
    frame = pd.read_parquet(path, columns=[ID, LABEL, *afm_columns, *ree_columns])
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("Expected one row per physical sample.")
    if set(frame[LABEL]) != set(GROUPS):
        raise ValueError("Unexpected set of tectonic classes.")
    oxides = frame[afm_columns].to_numpy(dtype=float)
    raw_ree = frame[list(ree_columns)].rename(columns=ree_columns).astype(float)
    if not np.isfinite(oxides).all() or (oxides < 0).any() or (oxides[:, 2:] <= 0).any():
        raise ValueError("Invalid AFM chemistry in the frozen cohort.")
    if not np.isfinite(raw_ree).all().all() or not raw_ree.gt(0).all().all():
        raise ValueError("REE concentrations must be finite and positive.")
    afm = close_afm(np.column_stack((oxides[:, 0] + oxides[:, 1], oxides[:, 2:])))
    np.testing.assert_allclose(afm.sum(axis=1), 1)
    normalized = raw_ree.div(PM)
    np.testing.assert_allclose(normalized.mul(PM), raw_ree)
    source = frame[[ID, LABEL, *afm_columns]].copy()
    source["F_Fe2O3T_wt_pct"] = oxides[:, 2] / FE_FACTOR
    source[["A_closed_pct", "F_closed_pct", "M_closed_pct"]] = afm * 100
    source = source.join(raw_ree.add_suffix("_ppm")).join(normalized.add_suffix("_PM"))
    return source, afm, normalized


def summarize(source, normalized):
    groups = []
    for group in GROUPS:
        values = normalized.loc[source[LABEL].eq(group)]
        stats = values.quantile(list(QUANTILES)).rename(index=QUANTILES).T
        stats.insert(0, "n", len(values))
        stats.insert(0, "element", stats.index)
        stats.insert(0, LABEL, group)
        groups.append(stats)
    return pd.concat(groups, ignore_index=True)


def afm_figure(source, afm):
    fig, axes = plt.subplots(3, 3, figsize=(7.205, 7.165))
    fig.subplots_adjust(left=0.040, right=0.96, bottom=0.075, top=0.89,
                        wspace=0.25, hspace=0.25)
    fig.text(0.04, 0.970, "AFM compositions by tectonic setting", fontsize=13, weight="bold")
    fig.text(0.04, 0.942, f"D36 expansion13  |  {len(source):,} physical samples",
             color="#526172", fontsize=8)
    triangle = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)])
    magnesia = np.linspace(BOUNDARY[0, 2], BOUNDARY[-1, 2], 180)
    iron = PchipInterpolator(BOUNDARY[:, 2], BOUNDARY[:, 1])(magnesia)
    boundary = close_afm(np.column_stack((100 - magnesia - iron, iron, magnesia)))
    for index, ((group, (name, color)), ax) in enumerate(zip(GROUPS.items(), axes.flat)):
        mask = source[LABEL].eq(group)
        ax.set(xlim=(-0.065, 1.065), ylim=(-0.13, 1.09), aspect="equal")
        ax.axis("off")
        for proportion in (0.25, 0.5, 0.75):
            for axis in range(3):
                edge = np.array([(proportion, 1 - proportion, 0),
                                 (proportion, 0, 1 - proportion)])
                ax.plot(*xy(np.roll(edge, axis, axis=1)), color="#DCE3EB", linewidth=0.45)
        ax.plot(*xy(triangle), color="#263444", linewidth=0.75)
        ax.scatter(*xy(afm[mask]), s=3.7, c=color, linewidths=0, alpha=0.24,
                   rasterized=True, zorder=3)
        ax.plot(*xy(boundary), color="#223142", linewidth=0.8, linestyle="--", zorder=4)
        ax.text(-0.05, 1.038, chr(97 + index), fontsize=8, weight="bold")
        ax.text(0.055, 1.038, f"{group}  {name}", color=color, fontsize=7.3, weight="bold")
        ax.text(1.02, 0.90, f"n = {mask.sum():,}", fontsize=6.3, ha="right", color="#526172")
        ax.text(0.5, 0.906, "Fe₂O₃(T)", fontsize=6.5, ha="center")
        ax.text(0, -0.041, "Na₂O + K₂O", fontsize=6.5, ha="left", va="top")
        ax.text(1, -0.041, "MgO", fontsize=6.5, ha="right", va="top")
    fig.legend(handles=[Line2D([0], [0], color="#223142", linestyle="--",
                               linewidth=0.9, label="Irvine–Baragar series boundary on the Fe₂O₃(T) basis")],
               loc="lower left", bbox_to_anchor=(0.038, 0.017), fontsize=7)
    return fig


def ree_figure(stats, sample_count):
    if (stats["p05"] <= 0).any():
        raise ValueError("Logarithmic plotting requires strictly positive whiskers.")
    fig, axes = plt.subplots(4, 3, figsize=(7.205, 8.150), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.097, right=0.969, bottom=0.112, top=0.900,
                        wspace=0.16, hspace=0.38)
    fig.text(0.04, 0.973, "Rare-earth element patterns", fontsize=13, weight="bold")
    fig.text(0.04, 0.949, f"D36 expansion13  |  {sample_count:,} physical samples",
             color="#526172", fontsize=8)
    fig.supylabel("Sample / primitive mantle", x=0.026, fontsize=8, y=0.51)
    x = np.arange(len(PM))
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.set(ylim=(0.65, 230), xlim=(-0.55, len(x) - 0.45))
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", linewidth=0.5, color="#DEE5ED")
        ax.axhline(1, color="#8794A2", linestyle=":", linewidth=0.7)
        ax.set_xticks(x, labels=PM.index)
        ax.tick_params(axis="x", labelsize=5.6, length=0)
        ax.tick_params(axis="y", labelsize=6, length=2)
    profiles = {group: stats.loc[stats[LABEL].eq(group)].set_index("element").loc[PM.index]
                for group in GROUPS}
    for index, ((group, (name, color)), ax) in enumerate(zip(GROUPS.items(), axes.flat)):
        profile = profiles[group]
        ax.fill_between(x, profile["p05"], profile["p95"], color=color, alpha=0.16, linewidth=0)
        boxes = [
            {**row[["q1", "q3"]].to_dict(), "med": row["median"],
             "whislo": row["p05"], "whishi": row["p95"]}
            for _, row in profile.iterrows()
        ]
        ax.bxp(boxes, positions=x, widths=0.52, manage_ticks=False,
               showfliers=False, showcaps=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor=color, linewidth=0.65),
               medianprops=dict(color=color, linewidth=1.0),
               whiskerprops=dict(color=color, linewidth=0.55))
        ax.plot(x, profile["median"], color=color, linewidth=0.9)
        ax.set_title(f"{chr(97 + index)}   {group}  {name}", loc="left", fontsize=6.9,
                     color=color, weight="bold", pad=5)
        ax.text(0.975, 0.94, f'n = {profile["n"].iloc[0]:,}', transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1, color="#526172")
    comparisons = [("BAB", "MOR", "IOA"), ("CA", "OI", "CFB"), ("CA", "IA", "IOA")]
    for index, (ax, groups) in enumerate(zip(axes[-1], comparisons), start=9):
        ax.set_title(f"{chr(97 + index)}   " + " / ".join(groups), loc="left",
                     fontsize=7, weight="bold", pad=5)
        for group, style in zip(groups, ("-", "--", ":")):
            color = GROUPS[group][1]
            profile = profiles[group]
            ax.fill_between(x, profile["q1"], profile["q3"], color=color, alpha=0.14, linewidth=0)
            ax.plot(x, profile["median"], color=color, linestyle=style, linewidth=1.1, label=group)
        ax.legend(loc="upper right", ncol=3, fontsize=5.6, handlelength=1.4,
                  columnspacing=0.6, handletextpad=0.3, borderaxespad=0.35)
    fig.text(0.097, 0.055, "a–i  Box Q1–Q3 · median line · whiskers and shading P5–P95",
             color="#526172", fontsize=6.6)
    fig.text(0.097, 0.038, "j–l  Median lines and Q1–Q3 shading   |   McDonough & Sun (1995), Table 5",
             color="#526172", fontsize=6.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.65,
        "legend.frameon": False, "text.color": "#223142",
    })
    source, afm, normalized = load_data(args.data)
    stats = summarize(source, normalized)
    if not stats["p05"].between(0.65, 230).all() or not stats["p95"].between(0.65, 230).all():
        raise ValueError("Expand the common REE y-axis to show all plotted whiskers.")
    source.to_csv(args.out / "figure_source_data.csv.gz", index=False)
    stats.to_csv(args.out / "ree_summary.csv", index=False)
    PM.rename_axis("element").to_csv(args.out / "primitive_mantle_ppm.csv")
    for name, fig in (("AFM", afm_figure(source, afm)), ("REE_spider_box", ree_figure(stats, len(source)))):
        fig.savefig(args.out / f"{name}.png", dpi=300, facecolor="white")
        fig.savefig(args.out / f"{name}.pdf", dpi=600, facecolor="white")
        fig.savefig(args.out / f"{name}.svg", dpi=600, facecolor="white")
        fig.savefig(args.out / f"{name}.tiff", dpi=600, facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    print(f"Input SHA-256 {hashlib.sha256(args.data.read_bytes()).hexdigest()}")
    print(stats.pivot(index=LABEL, columns="element", values="median")[["La", "Sm", "Yb"]])


if __name__ == "__main__":
    main()
