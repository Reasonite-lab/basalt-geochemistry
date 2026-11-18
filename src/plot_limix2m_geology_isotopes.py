"""Plot paired accuracy changes and supply audited isotope values to the revision."""

import json

import numpy as np
import pandas as pd

from plot_limix2m_geology_paper import BLUE, RUST, GRAY, OUT, export, panel, plt, style
from run_limix2m_geology_isotopes import BASE, MAIN, PB_FLAGS, PB_VALUES, SRND


RESULTS = OUT.with_name("limix2m_geology_isotopes_2026-11-17")
FIGURE = "图6_同位素输入对照"
MODELS = {"LimiX-2M": BLUE, "LightGBM": RUST}


def build_evidence():
    audit = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS / "scores_summary.csv")
    by_seed = pd.read_csv(RESULTS / "scores_by_seed.csv")
    delta = pd.read_csv(RESULTS / "paired_deltas_by_seed.csv")
    assert audit["status"] == "PASS" and len(audit["seeds"]) == 5 and audit["budget"] == 2048
    assert summary.splits.eq(5).all()
    pairs = by_seed[by_seed.scenario.eq(MAIN)].pivot(
        index=["seed", "subset", "model"], columns="panel", values="accuracy")
    evidence = {
        "cohort": audit["cohort_isotope_counts"],
        "summary": summary.to_dict(orient="records"),
        "srnd_gain_splits": int((pairs.xs(("全部", "LimiX-2M"), level=("subset", "model"))[SRND] >
                                 pairs.xs(("全部", "LimiX-2M"), level=("subset", "model"))[BASE]).sum()),
    }
    (RESULTS / "manuscript_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return pairs, delta


def isotope_figure(pairs, delta):
    style()
    dest = RESULTS / "figures_zh"
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.13))
    fig.subplots_adjust(left=.17, right=.965, top=.8, bottom=.24, wspace=.64)
    a, b = axes
    labels_a = ("加 Sr–Nd", "加 Sr–Nd、Pb 标记", "加 Sr–Nd、Pb 实测")
    labels_b = ("相对于 Sr–Nd", "相对于 Pb 标记")
    changes = [[], []]
    for (model, color), offset in zip(MODELS.items(), (-.13, .13), strict=True):
        all_test = pairs.xs(("全部", model), level=("subset", "model"))
        pb_delta = delta[delta.scenario.eq(MAIN) & delta.subset.eq("至少一项Pb") &
                         delta.model.eq(model) & delta.target.eq(PB_VALUES)]
        values_a = [all_test[target] - all_test[BASE] for target in (SRND, PB_FLAGS, PB_VALUES)]
        values_b = [pb_delta.loc[pb_delta.reference.eq(ref), "accuracy"] for ref in (SRND, PB_FLAGS)]
        for j, (ax, values) in enumerate(zip(axes, (values_a, values_b), strict=True)):
            for i, values_for_row in enumerate(values):
                assert len(values_for_row) == 5
                x = values_for_row.to_numpy() * 100
                changes[j].extend(x)
                y = i + offset
                ax.scatter(x, np.full_like(x, y), s=12, color=color, alpha=.36, linewidth=0)
                ax.scatter(x.mean(), y, s=28, color=color, edgecolors="white", linewidth=.45, marker="D",
                           label=model if i == 0 else None, zorder=3)
    lo = float(np.floor(min(map(min, changes))))
    hi = float(np.ceil(max(map(max, changes))))
    for ax, labels, letter, title in zip(axes, (labels_a, labels_b), "ab",
                                          ("全部测试样品", "有 Pb 实测值的测试样品"), strict=True):
        ax.set_yticks(range(len(labels)), labels)
        ax.set(ylim=(len(labels) - .45, -.55), xlim=(lo, hi), xlabel="配对准确率差（百分点）")
        ax.axvline(0, ls="--", linewidth=.65, color=GRAY)
        ax.grid(axis="x", color=GRAY, alpha=.19)
        ax.set_axisbelow(True)
        panel(ax, letter, title)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.57, .985), ncol=2)
    fig.text(.075, .063, "圆点为各次划分的差值，菱形为五次均值。a 以纯化学面板为参照，b 比较加入实测 Pb 的变化。",
             fontsize=7, color=GRAY)
    export(fig, FIGURE, dest)


if __name__ == "__main__":
    isotope_figure(*build_evidence())
