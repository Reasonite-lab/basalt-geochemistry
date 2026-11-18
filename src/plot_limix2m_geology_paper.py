"""Chinese-language figures for the basalt geochemistry evaluation."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from run_limix2m_geology_paper import BUDGETS, MASKS, OUT
from run_d36_srnd_partial_chemistry_2026_11_13 import E36, ID, LABEL, load_cohort


LABELS = {
    "BAB": "弧后盆地", "CA": "大陆弧", "CFB": "大陆溢流", "CR": "大陆裂谷",
    "IA": "岛弧", "IOA": "洋内弧", "MOR": "洋中脊", "OI": "大洋岛", "OP": "大洋高原",
}
SHORT = ["弧后", "大陆弧", "大陆溢流", "大陆裂谷", "岛弧", "洋内弧", "洋中脊", "大洋岛", "大洋高原"]
BLUE = "#1E6480"
RUST = "#9D5637"
GRAY = "#60727A"
SLATE = "#21333D"
MODELS = {"LimiX-2M": BLUE, "LightGBM": RUST}


def style():
    font_path = "C:/Windows/Fonts/Noto Sans SC (TrueType).otf"
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"], "font.size": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "axes.unicode_minus": False,
        "legend.frameon": False, "text.color": SLATE, "axes.labelcolor": SLATE,
    })


def export(fig, name, dest):
    fig.savefig(dest / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(dest / f"{name}.pdf", dpi=600, facecolor="white")
    fig.savefig(dest / f"{name}.svg", dpi=600, facecolor="white")
    fig.savefig(dest / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9, weight="bold", pad=8)


def data_figure(dest):
    cohort, x, _, _ = load_cohort()
    fig, axs = plt.subplots(1, 2, figsize=(7.205, 3.46), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.124, right=.975, top=.876, bottom=.218, wspace=.38)
    ax = axs[0]
    counts = pd.crosstab(cohort[LABEL], cohort.D36_complete).loc[list(LABELS)]
    counts.to_csv(dest / "图1_分类型样品数.csv")
    order = np.arange(len(LABELS))
    ax.barh(order, counts[True], height=.67, color=BLUE, label="D36 齐全")
    ax.barh(order, counts[False], height=.67, color="#A8C2C9", left=counts[True],
            label="D36 部分缺测")
    ax.set_yticks(order, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlabel("实体样品数")
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper right")
    panel(ax, "a", "共同22项队列的测量覆盖")
    ax = axs[1]
    colors = {"BAB": "#77568D", "MOR": BLUE, "IOA": "#338269"}
    ix = {element: i for i, element in enumerate(E36)}
    # Input contains log concentrations; both axes are log10 element ratios.
    x_ratio = x[:, ix["NB(PPM)"]] - x[:, ix["YB(PPM)"]]
    y_ratio = x[:, ix["TH(PPM)"]] - x[:, ix["YB(PPM)"]]
    frame = pd.DataFrame({ID: cohort[ID], LABEL: cohort[LABEL], "log10_Nb_Yb": x_ratio,
                          "log10_Th_Yb": y_ratio})
    frame = frame[frame[LABEL].isin(colors)]
    frame.to_csv(dest / "图1_比值源数据.csv.gz", index=False)
    coverage = {}
    for name, color in colors.items():
        class_rows = frame[frame[LABEL].eq(name)]
        cloud = class_rows.dropna()
        coverage[name] = {"全部": len(class_rows), "三项均实测": len(cloud)}
        ax.scatter(cloud.log10_Nb_Yb, cloud.log10_Th_Yb, s=4, alpha=.22, linewidths=0,
                   color=color, rasterized=True, label=f"{LABELS[name]}（{len(cloud):,}）")
        ax.scatter(cloud.log10_Nb_Yb.median(), cloud.log10_Th_Yb.median(),
                   marker="D", color=color, edgecolor="white", s=20, linewidth=.6, zorder=5)
    pd.DataFrame.from_dict(coverage, orient="index").to_csv(dest / "图1_比值覆盖统计.csv")
    ax.set(xlabel="log10（Nb/Yb）", ylabel="log10（Th/Yb）")
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right", markerscale=2)
    panel(ax, "b", "弧后、洋脊与洋内弧的比值分布")
    fig.text(.124, .047, "菱形为各组中位数。散点包含三项比值组分均有实测记录的样品。", fontsize=7, color=GRAY)
    export(fig, "图1_样品覆盖与地球化学比值", dest)


def budget_figure(scores, dest):
    s = scores[scores.scenario.eq("原始测试") & scores.subset.eq("全部") &
               scores.model.isin([*MODELS, "全量LightGBM"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.205, 3.45), sharex=True)
    fig.subplots_adjust(left=.091, right=.970, top=.793, bottom=.19, wspace=.29)
    for ax, metric, title, letter in zip(axs, ("accuracy", "macro_f1_9_labels"),
                                         ("总体准确率（%）", "宏平均 F1（%）"), "ab"):
        for model, c in MODELS.items():
            for panel_name, pattern, marker in [("D22_42", "--", "s"), ("D36_74", "-", "o")]:
                val = s[s.model.eq(model) & s.panel.eq(panel_name)].groupby("budget")[metric].agg(["mean", "std"])
                ax.errorbar(np.arange(3), val.loc[list(BUDGETS), "mean"] * 100,
                            yerr=val.loc[list(BUDGETS), "std"] * 100, color=c, linestyle=pattern,
                            marker=marker, markersize=3.5, capsize=2, linewidth=1.25,
                            label=f"{model} · {panel_name}")
        ref = s[s.model.eq("全量LightGBM") & s.panel.eq("D36_74")][metric].mean()
        ax.axhline(ref * 100, color=GRAY, linewidth=.85, linestyle=":",
                   label="全量 LightGBM · D36_74")
        ax.set(xticks=np.arange(3), xticklabels=[f"{v:,}" for v in BUDGETS], xlabel="带标签样品数")
        ax.grid(axis="y", alpha=.25)
        panel(ax, letter, title)
    handles, keys = axs[0].get_legend_handles_labels()
    fig.legend(handles, keys, loc="upper center", bbox_to_anchor=(.55, .985), ncol=3, fontsize=7)
    fig.text(.091, .042, "点与误差线为五次固定随机划分的均值和标准差。全量树每次使用 19,400 件训练样品。",
             fontsize=7, color=GRAY)
    export(fig, "图2_化学面板与标注预算", dest)


def masks_figure(scores, dest):
    s = scores[scores.scenario.isin(MASKS) & scores.subset.eq("全部") & scores.model.isin(MODELS)]
    fig, axs = plt.subplots(1, 2, figsize=(7.205, 3.35))
    fig.subplots_adjust(left=.09, right=.961, top=.836, bottom=.19, wspace=.28)
    for ax, metric, title, letter, factor in zip(axs, ("accuracy","log_loss"),
                                                  ("准确率（%）","多类对数损失"), "ab", (100, 1)):
        for name, color in MODELS.items():
            metrics = s[s.model.eq(name)].groupby("scenario")[metric].agg(["mean", "std"]).loc[list(MASKS)]
            ax.errorbar(np.arange(3), metrics["mean"] * factor, yerr=metrics["std"] * factor,
                        label=name, color=color, linewidth=1.35, marker="o", markersize=4, capsize=3)
        ax.set(xticks=np.arange(3), xticklabels=["全部实测","等量随机遮挡","实测组合遮挡"],
               xlim=(-.24, 2.24))
        ax.grid(axis="y", alpha=.25)
        panel(ax, letter, title)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.50, .985), ncol=2)
    fig.text(.09, .06, "两种遮挡对每件样品遮去相同测量数。点和误差线为五次划分的均值和标准差。",
             fontsize=7, color=GRAY)
    export(fig, "图3_测量组合遮挡对照", dest)


def confusion_figure(dest):
    confusion = pd.read_csv(OUT / "confusion.csv")
    c = confusion[confusion.panel.eq("D36_74") & confusion.budget.eq(2048) &
                  confusion.scenario.eq("原始测试") & confusion.subset.eq("全部") & confusion.model.isin(MODELS)]
    fig, axs = plt.subplots(1, 2, figsize=(7.205, 3.56))
    fig.subplots_adjust(left=.086, right=.902, bottom=.238, top=.84, wspace=.35)
    for ax, name, letter in zip(axs, MODELS, "ab"):
        a = c[c.model.eq(name)].pivot_table(index="true", columns="predicted", values="count", aggfunc="sum")
        a = a.reindex(index=LABELS, columns=LABELS).to_numpy()
        pct = 100 * a / a.sum(1, keepdims=True)
        mesh = ax.imshow(pct, vmin=0, vmax=100, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(SHORT)), SHORT, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(SHORT)), SHORT)
        ax.tick_params(length=0, labelsize=7)
        for i, j in np.ndindex(pct.shape):
            ax.text(j, i, f"{pct[i, j]:.0f}", va="center", ha="center", size=6.2,
                    color="white" if pct[i, j] > 54 else SLATE)
        ax.set_xlabel("预测构造类别")
        panel(ax, letter, name)
    axs[0].set_ylabel("记录构造类别")
    bar = fig.colorbar(mesh, cax=fig.add_axes([.906, .28, .017, .49]), ticks=[0, 25, 50, 75, 100])
    bar.ax.tick_params(labelsize=7)
    bar.ax.set_title("行比例\n（%）", fontsize=7, loc="left", pad=8)
    fig.text(.087, .039, "共同22项队列，D36_74，2,048 件带标签样品。汇总五次测试记录后逐行归一化。",
             fontsize=7, color=GRAY)
    export(fig, "图4_九类构造判别混淆", dest)


def expansion_figure(dest):
    s = pd.read_csv(OUT / "expansion/scores.csv")
    s = s[s.context.eq("mixed") & s.model.isin([*MODELS, "LightGBM校准"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.205, 3.22))
    fig.subplots_adjust(left=.085, right=.968, top=.835, bottom=.18, wspace=.29)
    for ax, metric, title, letter, factor in zip(axs, ("accuracy","log_loss"),
                                                  ("准确率（%）","多类对数损失"), "ab", (100, 1)):
        for name, c in {"LimiX-2M": BLUE, "LightGBM": RUST, "LightGBM校准": "#BB927B"}.items():
            if name.endswith("校准") and metric == "accuracy":
                continue
            t = s[s.model.eq(name)].set_index("subset").loc[["全部", "D36齐全", "D36部分缺测"], metric]
            ax.plot(np.arange(3), t.to_numpy() * factor, marker="o", markersize=4,
                    color=c, linestyle="--" if name.endswith("校准") else "-", label=name)
        ax.set(xticks=np.arange(3), xticklabels=["全部样品","完整 D36","真实缺测"],
               xlim=(-.19, 2.19))
        ax.grid(axis="y", alpha=.25)
        panel(ax, letter, title)
    handles, names = axs[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .985), ncol=3)
    fig.text(.085, .045, "扩展队列共 49,676 件。混合上下文含 2,048 件样品，结果合并随机五折。",
             fontsize=7, color=GRAY)
    export(fig, "图5_真实缺测队列的模型比较", dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    dest = OUT / "figures_zh"
    dest.mkdir(parents=True, exist_ok=True)
    style()
    data_figure(dest)
    if not args.static_only:
        scores = pd.read_csv(OUT / "scores.csv")
        budget_figure(scores, dest)
        masks_figure(scores, dest)
        confusion_figure(dest)
        expansion_figure(dest)


if __name__ == "__main__":
    main()
