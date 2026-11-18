"""Publication figures for recorded LightGBM fitting and fixed-panel masking."""

import numpy as np
import pandas as pd

from plot_limix2m_geology_paper import BLUE, GRAY, RUST, OUT, export, panel, plt, style

DATA = OUT.with_name("limix2m_panel_control_2026-11-17")
GROUPS = ("实测组合遮挡", "扩展14项随机遮挡", "随机遮挡")
TITLES = ("真实组合", "扩展14项内随机", "全36项内随机")


def learning_curve():
    history = pd.read_csv(DATA / "tree_training_history.csv")
    seed = history.seed.min()
    current = history[history.seed.eq(seed)]
    rounds, = current.selected_round.unique()
    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.05))
    fig.subplots_adjust(left=.105, right=.975, top=.79, bottom=.205, wspace=.32)
    for ax, metric, factor, letter, title in zip(
        axes, ("loss", "accuracy"), (1, 100), "ab", ("多类对数损失", "分类准确率（%）"), strict=True
    ):
        for name, color, label in (("train", BLUE, "内部拟合集"), ("validation", RUST, "内部验证集")):
            d = current[current.subset.eq(name)]
            ax.plot(d["round"], d[metric] * factor, color=color, lw=1.35, label=label)
            value = d.loc[d["round"].eq(rounds), metric].item()
            ax.scatter(rounds, value * factor, color=color, s=13, zorder=3)
        ax.axvline(rounds, color=GRAY, linestyle="--", linewidth=.8)
        ax.grid(axis="y", alpha=.2)
        ax.set_xlabel("提升轮数")
        panel(ax, letter, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.text(.105, .053, f"首次划分，2,048 件训练侧样品用于选轮。虚线为验证损失最小时的第 {rounds} 轮。",
             color=GRAY, fontsize=7)
    export(fig, "训练曲线", DATA)


def paired_mask():
    scores = pd.read_csv(DATA / "paired_mask_scores.csv")
    accuracy = scores.pivot(index=["seed", "model"], columns="scenario", values="accuracy")
    fig, ax = plt.subplots(figsize=(7.205, 3.15))
    fig.subplots_adjust(left=.235, right=.955, top=.80, bottom=.25)
    for model, c, offset in (("LimiX-2M", BLUE, -.13), ("LightGBM", RUST, .13)):
        d = accuracy.xs(model, level="model")
        for j, group in enumerate(GROUPS):
            drop = 100 * (d[group]-d["完整实测"])
            assert len(drop) == 5
            row = j + offset
            ax.scatter(drop, np.full(len(drop), row), s=20, color=c, linewidth=0, alpha=.35)
            ax.scatter(drop.mean(), row, marker="D", s=38, color=c,
                       edgecolors="white", linewidth=.6, label=model if j == 0 else None)
    ax.axvline(0, color=GRAY, linewidth=.8, linestyle="--")
    ax.set(yticks=range(3), yticklabels=TITLES, ylim=(2.55,-.55),
           xlabel="相对完整实测的准确率变化（百分点）")
    ax.grid(axis="x", alpha=.22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.text(.095, .065, "浅色点为各次划分，菱形为均值。每次使用同一批完整测试样品与相同的遮挡项数量。",
             color=GRAY, fontsize=7)
    export(fig, "固定基础测项的遮挡对照", DATA)


if __name__ == "__main__":
    style()
    learning_curve()
    paired_mask()
