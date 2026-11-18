"""Replot the earlier TAS and AFM classifications for the current main cohort."""

import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pyrolite.util.classification import TAS, _read_poly, get_centroid
from scipy.interpolate import PchipInterpolator

from plot_limix2m_geology_paper import LABELS, export, plt, style
from run_limix2m_geology_paper import E36, ID, LABEL, ROOT, load_cohort

sys.path.insert(0, str(ROOT / "scripts"))
from plot_doucet_afm_ree import BOUNDARY, close_afm, xy


OUT = ROOT / "figures/current_basalt_classification_2026-11-18"
COLORS = dict(zip(LABELS, ("#80549B", "#156946", "#AB402E", "#8A5F2D", "#4F7836",
                          "#147885", "#17638E", "#A43D74", "#4E577E"), strict=True))
FIELDS = {"Pc": "苦橄岩", "Bs": "亚碱性玄武岩", "Ba": "碱性玄武岩",
          "O1": "玄武安山岩", "O2": "安山岩", "S1": "粗面玄武岩",
          "S2": "玄武质粗面安山岩", "U1": "碱玄岩", "U2": "响岩质碱玄岩"}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    style()
    cohort, features, _, _ = load_cohort()
    raw = np.power(10, features[:, :36].astype(float))
    assert len(cohort) == 24251 and np.isfinite(raw[:, :10]).all()
    pos = {name: i for i, name in enumerate(E36)}
    major = 100 * raw[:, :10] / raw[:, :10].sum(axis=1, keepdims=True)
    silica = major[:, pos["SIO2(WT%)"]]
    alkali = major[:, pos["NA2O(WT%)"]] + major[:, pos["K2O(WT%)"]]
    afm = close_afm(np.column_stack((
        raw[:, pos["NA2O(WT%)"]] + raw[:, pos["K2O(WT%)"]],
        raw[:, pos["FEOT(WT%)"]], raw[:, pos["MGO(WT%)"]])))
    source = cohort[[ID, LABEL]].copy()
    source["SiO2_anhydrous_pct"], source["alkalis_anhydrous_pct"] = silica, alkali
    source[["A_pct", "F_Fe2O3T_pct", "M_pct"]] = 100 * afm
    source.to_csv(OUT / "分类图源数据.csv.gz", index=False)
    cohort.groupby(LABEL).size().to_csv(OUT / "各类样品数.csv")

    fig, ax = plt.subplots(figsize=(7.205, 4.4))
    fig.subplots_adjust(left=.105, right=.975, top=.88, bottom=.16)
    classifier = TAS(which_model="LeMaitre")
    classifier.add_to_axes(ax=ax, fill=False, add_labels=False, axes_scale=100,
                           edgecolor="#34444F", linewidth=.65, zorder=3)
    for name, color in COLORS.items():
        mask = cohort[LABEL].eq(name)
        ax.scatter(silica[mask], alkali[mask], s=3, alpha=.19, color=color, linewidths=0,
                   label=LABELS[name], rasterized=True, zorder=2)
        ax.scatter(np.median(silica[mask]), np.median(alkali[mask]), marker="D", s=21,
                   color=color, edgecolors="white", linewidths=.55, zorder=5)
    for field, text in FIELDS.items():
        polygon = Polygon(np.asarray(_read_poly(classifier.fields[field]["poly"]), float))
        x, y = get_centroid(polygon)
        if 40 <= x <= 60 and 0 <= y <= 12:
            ax.text(x, y, text, fontsize=7, ha="center", va="center", zorder=4, color="#1E333F")
    ax.set(xlim=(40, 60), ylim=(0, 12), xlabel="SiO2（无水归一化，质量分数 %）",
           ylabel="Na2O + K2O（无水归一化，质量分数 %）")
    ax.grid(alpha=.18, linewidth=.5)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=c, markersize=4,
                      label=LABELS[name]) for name, c in COLORS.items()]
    fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(.54, 1), fontsize=7)
    inside = (silica >= 40) & (silica <= 60) & (alkali >= 0) & (alkali <= 12)
    pd.Series({"全部样品": len(cohort), "显示范围内": int(inside.sum()),
               "显示范围外": int((~inside).sum())}).to_csv(OUT / "TAS显示范围统计.csv")
    fig.text(.105, .035, f"24,251 件样品。菱形为分量中位数，图示范围内 {inside.sum():,} 件。", fontsize=7)
    export(fig, "TAS分类图", OUT)

    fig, axes = plt.subplots(3, 3, figsize=(7.205, 5.7))
    fig.subplots_adjust(left=.04, right=.965, top=.935, bottom=.07, wspace=.25, hspace=.28)
    triangle = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)])
    m = np.linspace(BOUNDARY[0, 2], BOUNDARY[-1, 2], 180)
    f = PchipInterpolator(BOUNDARY[:, 2], BOUNDARY[:, 1])(m)
    boundary = close_afm(np.column_stack((100-m-f, f, m)))
    for i, ((name, color), ax) in enumerate(zip(COLORS.items(), axes.flat, strict=True)):
        mask = cohort[LABEL].eq(name)
        ax.set(xlim=(-.06, 1.06), ylim=(-.105, 1.035), aspect="equal")
        ax.axis("off")
        for proportion in (.25, .5, .75):
            edge = np.array([(proportion, 1-proportion, 0), (proportion, 0, 1-proportion)])
            for axis in range(3):
                ax.plot(*xy(np.roll(edge, axis, axis=1)), color="#DCE3EB", linewidth=.4)
        ax.plot(*xy(triangle), color="#263444", lw=.7)
        ax.scatter(*xy(afm[mask]), s=2.2, color=color, alpha=.24, linewidths=0, rasterized=True)
        ax.plot(*xy(boundary), color="#223142", lw=.7, linestyle="--")
        ax.set_title(f"{chr(97+i)}  {LABELS[name]}（{mask.sum():,}）", fontsize=8, color=color, loc="left", pad=3)
        ax.text(.5, .91, "F", fontsize=7, ha="center")
        ax.text(0, -.036, "A", fontsize=7, ha="left", va="top")
        ax.text(1, -.036, "M", fontsize=7, ha="right", va="top")
    fig.text(.06, .025, "A = Na2O + K2O，F = Fe2O3(T)，M = MgO；虚线为 Irvine–Baragar 系列分界。", fontsize=7)
    export(fig, "AFM分类图", OUT)


if __name__ == "__main__":
    main()
