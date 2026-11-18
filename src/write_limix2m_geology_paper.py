"""Build the Chinese manuscript from the audited basalt experiment results."""

import json
from pathlib import Path
from statistics import stdev

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/modeling/limix2m_geology_paper_2026-11-17"
ISOTOPE_RESULTS = RESULTS.with_name("limix2m_geology_isotopes_2026-11-17")
DELIVERY = ROOT / "delivery/limix2m_玄武岩_2026-11-17"
FIG_DIR = RESULTS / "figures_zh"
TITLE = "基于LimiX-2M的玄武岩全岩地球化学构造环境判别"
SUBTITLE = "元素测量覆盖、标注样品数量与同位素信息"
LM = "LimiX-2M"
GB = "LightGBM"
PANELS = ("D22_42", "D36_74")
BUDGETS = (512, 2048, 4096)
CLASS_NAMES = dict(zip(
    ("BAB", "CA", "CFB", "CR", "IA", "IOA", "MOR", "OI", "OP"),
    ("弧后盆地", "大陆弧", "大陆溢流玄武岩", "大陆裂谷", "岛弧", "洋内弧", "洋中脊", "大洋岛", "大洋高原"),
))
FIGURES = [
    ("图1_样品覆盖与地球化学比值", "图1  主队列测量覆盖及弧后盆地、洋中脊和洋内弧样品的微量元素比值。a，各构造类别按 D36 测量完整性统计。b，Th、Nb、Yb 三项均有实测值的散点，菱形标出各类分量中位数。"),
    ("图2_化学面板与标注预算", "图2  标注数量与测量面板对判别表现的影响。a，总体准确率。b，九类宏平均 F1。点与误差线表示五次库内划分的均值及标准差，19,400 件为全训练集 LightGBM。"),
    ("图3_测量组合遮挡对照", "图3  同批完整测试样品的测量遮挡试验。a，准确率。b，多类对数损失。每件样品的随机遮挡和实测组合遮挡删除相同数量的测量，误差线为五次划分的标准差。"),
    ("图4_九类构造判别混淆", "图4  主队列 D36_74、2,048 件预算的九类混淆矩阵。a，LimiX-2M。b，LightGBM。矩阵合并五次测试记录后按真实类别归一化。同一实体可在不同划分中重复进入测试集。"),
    ("图5_真实缺测队列的模型比较", "图5  扩展队列的随机五折聚合评估，采用 D36 浓度对数。a，总体准确率。b，多类对数损失。混合上下文含 2,048 件，由各类别的完整与部分缺测样品构成。另给出 LightGBM 温度校准后的损失。"),
    ("图6_同位素输入对照", "图6  同位素输入的配对准确率差。a，全部测试样品，各方案相对于 D36_74。b，有可用 Pb 的测试样品中，Pb 实测输入分别相对于 Sr–Nd 输入及加 Pb 标记的输入。正值表示加入所列变量后准确率升高。圆点为五次划分的配对差值，菱形为均值。"),
]
ISO_PANELS = {
    "D36_74": "D36_74",
    "D36_74+SrNd": "加 Sr–Nd",
    "D36_74+SrNd+Pb_flags": "加 Sr–Nd 与 Pb 标记",
    "D36_74+SrNd+Pb": "加 Sr–Nd 与 Pb 实测",
}


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}"


def unique(rows, **keys):
    result, = (row for row in rows if all(row[k] == v for k, v in keys.items()))
    return result


def make_article(evidence, isotopes):
    def main(panel, budget, model):
        return unique(evidence["main"], panel=panel, budget=budget, model=model)

    def group(panel, budget, scenario, model, subset="全部"):
        return unique(evidence["mean"], panel=panel, budget=budget, scenario=scenario, model=model, subset=subset)

    def exp(subset, context, model):
        return unique(evidence["expansion"], subset=subset, context=context, model=model)

    def cls(label, model):
        return unique(evidence["class"], **{"class": label, "model": model})

    def iso(panel, model=LM, scenario="原始测试", subset="全部"):
        return unique(isotopes["summary"], panel=panel, scenario=scenario, model=model, subset=subset)

    srnd = iso("D36_74+SrNd")
    pb_values = iso("D36_74+SrNd+Pb")
    pb_flags = iso("D36_74+SrNd+Pb_flags")
    pb_subset = [iso(p, subset="至少一项Pb") for p in tuple(ISO_PANELS)[1:]]
    coverage = isotopes["cohort"]
    l2, t2 = (main("D36_74", 2048, m) for m in (LM, GB))
    l4 = main("D36_74", 4096, LM)
    full = main("D36_74", 19400, "全量LightGBM")
    nat_l, nat_t = (exp("D36部分缺测", "mixed", m) for m in (LM, GB))
    diff = l2["accuracy_mean"] - t2["accuracy_mean"]
    better = sum(row["accuracy"] > 0 for row in evidence["matched_deltas"])
    mask_names = ("完整实测", "随机遮挡", "实测组合遮挡")
    mask_l = [group("D36_74", 2048, s, LM) for s in mask_names]
    mask_t = [group("D36_74", 2048, s, GB) for s in mask_names]
    flags = group("D36_74", 2048, "原始测量标记", "仅测量标记")
    bab_to = sorted(
        ((row["count"], row["predicted"]) for row in evidence["confusion"]
         if row["true"] == "BAB" and row["predicted"] != "BAB"),
        reverse=True,
    )[:2]
    bab_text = "、".join(f'{CLASS_NAMES[label]}（{count} 条）' for count, label in bab_to)

    base = f"""# {TITLE}

{SUBTITLE}

## 摘要

**目的** 全岩地球化学可为玄武岩构造环境的判别提供证据，但文献汇编中元素测量覆盖不一，带有可信构造标签的样品数量也常受限。本文检验表格基础模型 LimiX-2M 在有限标注和部分元素缺测条件下的判别表现。

**方法** 以具有 22 项共同测量的 24,251 件玄武岩为主队列，采用五次分层随机 80∶20 划分，比较 D22_42 和 D36_74 面板上相同标注预算的 LimiX-2M 和 LightGBM，另设全量 LightGBM。配对比较等量随机遮挡与实测组合遮挡，并在 2,048 件预算下对照 Sr–Nd 比值、Pb 实测值和 Pb 可用性标记。以 D36 浓度对数在 49,676 件扩展队列进行库内随机五折。

**结果** D36_74 面板下，2,048 件标注时 LimiX-2M 与 LightGBM 准确率分别为 {pct(l2['accuracy_mean'])}% 和 {pct(t2['accuracy_mean'])}%，全量 LightGBM 为 {pct(full['accuracy_mean'])}%。LimiX-2M 加入 Sr–Nd 后达到 {pct(srnd['accuracy'])}%，再加 Pb 实测值回落至 {pct(pb_values['accuracy'])}%。扩展队列中 33,293 件真实缺测样品的同预算准确率分别为 {pct(nat_l['accuracy'])}% 和 {pct(nat_t['accuracy'])}%。弧后盆地与洋中脊及洋内弧仍存在明显混淆。

**结论** 在本资料库的有限标注条件下，LimiX-2M 对全岩元素数据具有较好的筛查表现。测量覆盖、参考样品组成及类别间地球化学重叠共同影响判别表现。本文的测评范围为库内插值，构造标签仍需结合产状和区域地质复核。

**关键词** 玄武岩；构造环境；全岩地球化学；Sr–Nd–Pb 同位素；部分缺测；上下文学习

## 1 引言

玄武岩的主量和微量元素组成记录了源区性质、部分熔融及岩浆演化，为判别岩浆形成的构造背景提供了依据。经典判别图利用有限元素比值识别洋中脊、弧和板内玄武岩的地球化学差异。Pearce（2008）讨论了 Nb/Yb、Th/Yb 等比值对源区富集与俯冲组分解释的含义[1]。同一构造背景内部可包含不同源区与演化过程，类别边界也可能随样品汇编及参照区域改变。Vermeesch（2006）、Ueki 等（2018）和 Doucet 等（2022）分别从统计判别、多变量地球化学和全球数据汇编角度扩展了这一问题[2–4]。

在实际资料中，不同研究采用的微量元素分析组合和检出限存在差异。仅保留齐全样品将改变训练与应用对象的组成。直接把部分数据输入模型也会带来另一问题，即模型有时可以利用“哪些元素被测量”的规律推测文献标签。本文把实测值的信息和测量覆盖作为两个可观察因素，在相同测试样品上改变输入面板或进行遮挡，并以仅含测量标记的模型衡量缺测模式可携带的标签信息。

LimiX-2M 根据带标签的上下文样品对待判样品进行预测[5]，适合研究参考样品数量受限的情形。LightGBM 是全岩地球化学表格数据可采用的梯度提升树基线[6]。本文考察标注预算和元素测量组合对构造判别的影响，并用真实缺测样品与主要混淆类别说明使用边界。Sr–Nd–Pb 同位素提供源区组成与演化的信息，其测量在汇编中分布不均。为评估这部分信息的增益，本文在固定标注预算和样品条件下进一步比较实测比值与分析可用性标记。

## 2 数据与方法

### 2.1 样品、构造标签与测量面板

主队列取自项目冻结的玄武岩全岩汇编。按项目实体样品标识汇总并采用冻结队列的质控与二级构造标签。九类分别为弧后盆地（BAB）、大陆弧（CA）、大陆溢流玄武岩（CFB）、大陆裂谷（CR）、岛弧（IA）、洋内弧（IOA）、洋中脊（MOR）、大洋岛（OI）和大洋高原（OP）。主队列各样品均有 22 项共同测量，共 24,251 件，其中 16,295 件满足 D36 完整测量，7,956 件至少缺少一项 D36 测量。图1展示各类别的覆盖情况，并以可计算 Nb/Yb 和 Th/Yb 的样品说明三个类别的比值分布。比值图用于描述，模型按同一面板规范统一构建输入。

D22_42 包括 10 项主量氧化物和 12 项微量元素，以 22 项浓度的常用对数、9 个主量组分的等距对数比坐标及 11 个地球化学对数比值构成。D36_74 在该基础上容纳 36 项测量，具有 36 个浓度对数、9 个主量等距对数比坐标、27 个对数比值和两项对数异常指标。主要比值涵盖 Th/Nb、La/Nb 和 Zr/Y 等。面板项目、转换和顺序由冻结代码固定。若某衍生量所需的任何组分缺测，则该衍生量也记为缺测。缺测保留为 NaN，由各模型自身处理。两种元素面板的定义均基于全岩元素测量。

扩展队列放宽共同测量限制，保留具有至少 18 项 D36 测量和可用九类标签的实体样品，共 49,676 件，其中 16,383 件测齐 D36，33,293 件为真实缺测，覆盖 1,678 种测量组合。扩展队列与主队列来自同一项目资料汇编，存在样品重叠，承担对较宽测量覆盖条件的库内检验。

[[FIG1]]

### 2.2 分割、模型与标注预算

主队列沿用随机种子 20261113–20261117 的五次分层 80∶20 划分，每次 19,400 件训练样品、4,851 件测试样品。每个划分从训练侧按类别比例抽取嵌套的 512、2,048 和 4,096 件。两模型、两面板复用同一批标注样品和测试样品。预算指带标签的实体样品总数。LimiX-2M 使用官方预训练权重和四成员推理集成，当前资料的带标签样品以推理上下文输入，未对权重作梯度更新。LightGBM 使用相同预算内的内部验证样品对多类对数损失早停，其后按选定轮数用完整预算重新拟合。全量 LightGBM 采用同样流程，使用每个划分的全部训练样品。

温度校准使用预算内独立的内部验证预测估计一个温度参数，再应用于该预算最终预测。树模型训练轮数与两模型温度均由训练侧确定。补充稳定性试验固定主队列第一个测试集，在每个标注预算和面板另行抽取两套上下文。扩展队列使用既定互斥测试折，并把具有相同化学数值指纹的记录放在同一折。两模型统一输入 36 项浓度的常用对数，保留缺测。每个训练折从完整样品中按类分层抽取 2,048 件作为对照参考集。混合参考集在各类内随机将至多一半完整样品替换成部分测量样品，保持参考集类别频数一致。

### 2.3 遮挡对照与指标

对主队列每个测试集的原本完整样品比较三种输入。第一种保留全部测量，第二种等量随机遮挡，第三种按训练侧缺测样品的实测组合进行遮挡。同一件测试样品在两种遮挡下减少相同测量数，遮挡后重新计算输入，参考样品仍为该次划分的同一 2,048 件。采用无构造标签参与匹配的训练侧供体测量组合。另以 D36 的 36 个有无测量标记训练 LightGBM，以衡量实际测量模式自身的预测信息。该对照分别应用于模拟遮挡输入和未改动的实际测试输入。

准确率统计所有测试记录的正确比例，宏平均 F1 等权汇总九类类内 F1。对数损失考察预测概率在真实类别上的分配，多类 Brier 分数计算各类别概率误差平方和，ECE 为按最大预测概率分十组的校准误差。主结果用五次划分的均值和标准差或汇总测试记录说明。扩展结果由每件样品的折外预测合并计算。五次重复划分之间测试样品可能重叠，因此以描述性比较为主。

### 2.4 同位素增量试验

按实体样品标识将同位素质控记录与主队列对齐，使用 A、B 级 ⁸⁷Sr/⁸⁶Sr、¹⁴³Nd/¹⁴⁴Nd、²⁰⁶Pb/²⁰⁴Pb、²⁰⁷Pb/²⁰⁴Pb 和 ²⁰⁸Pb/²⁰⁴Pb 实测比值。可用 Sr、Nd 和至少一种 Pb 比值的样品分别为 {coverage['Sr']:,}、{coverage['Nd']:,} 和 {coverage['any_Pb']:,} 件。比值沿用质控表中的实测记录，未经统一地质年龄校正。缺测或未通过质控的比值保留为 NaN。

以 D36_74 为基线，依次比较加入 Sr–Nd、在 Sr–Nd 基础上加入三个 Pb 实测值，以及仅加入三个对应的 Pb 可用性标记。四种输入共用每次划分的 2,048 件标注样品与 4,851 件测试样品。对全部测试样品和有可用 Pb 的同一子集分别统计，对数损失基于原始预测概率。元素遮挡复算沿用原来的遮挡位置，保留每件样品已有的同位素信息。

## 3 结果

### 3.1 测量面板与标注预算

在两种面板各三个同预算比较中，LimiX-2M 的准确率优势出现在 {better}/30 个“划分×面板×预算”配对中（表1，图2）。D36_74、2,048 件时，准确率平均差为 {pct(diff)} 个百分点，LimiX-2M 的宏平均 F1 为 {l2['macro_f1_9_labels_mean']:.3f}，LightGBM 为 {t2['macro_f1_9_labels_mean']:.3f}。在相同样品和标签条件下，LimiX-2M 使用 D36_74 较 D22_42 的平均准确率分别提高 {pct(main('D36_74', 512, LM)['accuracy_mean'] - main('D22_42', 512, LM)['accuracy_mean'])}、{pct(l2['accuracy_mean'] - main('D22_42', 2048, LM)['accuracy_mean'])} 和 {pct(l4['accuracy_mean'] - main('D22_42', 4096, LM)['accuracy_mean'])} 个百分点。D36_74 的全量 LightGBM 达到 {pct(full['accuracy_mean'])}% 和 {full['macro_f1_9_labels_mean']:.3f}，分别高于 4,096 件预算的 LimiX-2M。

表1  同预算模型和全量树模型的五次库内验证

|测量面板|标注数|模型|准确率均值±标准差（%）|宏平均 F1|对数损失|
|---|---:|---|---:|---:|---:|
"""
    for panel in PANELS:
        for n in (*BUDGETS, 19400):
            for model in (LM, GB) if n != 19400 else ("全量LightGBM",):
                r = main(panel, n, model)
                base += (
                    f"|{panel}|{n:,}|{model}|{pct(r['accuracy_mean'])} ± {pct(r['accuracy_std'])}|"
                    f"{r['macro_f1_9_labels_mean']:.3f}|{r['log_loss_mean']:.3f}|\n"
                )
    base += f"""
[[FIG2]]

### 3.2 同一样品的测量遮挡试验

完整实测、等量随机遮挡和实测组合遮挡下，LimiX-2M 的准确率依次为 {"、".join(pct(m['accuracy']) + '%' for m in mask_l)}，LightGBM 依次为 {"、".join(pct(m['accuracy']) + '%' for m in mask_t)}（图3）。对应对数损失在 LimiX-2M 为 {"、".join(f"{m['log_loss']:.3f}" for m in mask_l)}，在 LightGBM 为 {"、".join(f"{m['log_loss']:.3f}" for m in mask_t)}。等量随机遮挡时，LimiX-2M 的准确率低于同预算树模型。未改动测试集上，仅使用 36 个实测标记得到 {pct(flags['accuracy'])}% 的准确率与 {flags['macro_f1_9_labels']:.3f} 的宏平均 F1。仅预测训练集最多见的大陆溢流类时，汇总测试记录的准确率为 {pct(cls('CFB', LM)['n_class'] / (5 * 4851))}%，表明实际资料中的测量组合本身包含构造标签关联信息。

[[FIG3]]

### 3.3 构造类别与误判分布

主队列 D36_74 的 2,048 件预算下，LimiX-2M 对弧后盆地、大洋高原、大陆裂谷的召回率分别为 {pct(cls('BAB', LM)['recall'])}%、{pct(cls('OP', LM)['recall'])}% 和 {pct(cls('CR', LM)['recall'])}%（表2，图4）。弧后盆地未判回记录中，较多被判成{bab_text}。该现象须与图1所示范围一起读。图1只绘出比值各组分实测的子集，反映类间分布重叠及同类变化，无法将逐条误判归因到单个比值。

表2  D36_74 面板 2,048 件预算的分类型判别

|构造类别|测试记录数|LimiX-2M 召回率（%）|LightGBM 召回率（%）|LimiX-2M F1|LightGBM F1|
|---|---:|---:|---:|---:|---:|
"""
    for label, name in CLASS_NAMES.items():
        a, b = (cls(label, m) for m in (LM, GB))
        base += f"|{name}（{label}）|{a['n_class']:,}|{pct(a['recall'])}|{pct(b['recall'])}|{a['class_f1']:.3f}|{b['class_f1']:.3f}|\n"
    base += f"""
表注  记录数和召回率由五次划分合并得到，F1 为五次划分的均值。

[[FIG4]]

### 3.4 扩展队列的真实缺测与概率表现

在 49,676 件扩展队列的随机五折中，混合上下文下 LimiX-2M 的总体准确率为 {pct(exp('全部', 'mixed', LM)['accuracy'])}%，对数损失为 {exp('全部', 'mixed', LM)['log_loss']:.3f}。真实缺测子集中，两模型准确率分别为 {pct(nat_l['accuracy'])}% 和 {pct(nat_t['accuracy'])}%，宏平均 F1 为 {nat_l['macro_f1_9_labels']:.3f} 和 {nat_t['macro_f1_9_labels']:.3f}，对数损失为 {nat_l['log_loss']:.3f} 和 {nat_t['log_loss']:.3f}（图5）。当上下文只用完整样品，LimiX-2M 对真实缺测子集的准确率为 {pct(exp('D36部分缺测', 'complete', LM)['accuracy'])}%，LightGBM 为 {pct(exp('D36部分缺测', 'complete', GB)['accuracy'])}%。两种上下文保持相同的类别频数，同时改变了测量覆盖和所用实体样品。这一差异反映参考集选择的综合影响。扩展试验采用 D36 浓度输入，宜与主队列 D36_74 的结果分别解读。

主队列 D36_74、2,048 件下，温度校准前后 LimiX-2M 对数损失由 {l2['log_loss_mean']:.3f} 变为 {main('D36_74', 2048, 'LimiX-2M校准')['log_loss_mean']:.3f}，LightGBM 由 {t2['log_loss_mean']:.3f} 变为 {main('D36_74', 2048, 'LightGBM校准')['log_loss_mean']:.3f}。原始 ECE 分别为 {l2['ece_mean']:.3f} 和 {t2['ece_mean']:.3f}，原始多类 Brier 分数分别为 {l2['brier_mean']:.3f} 和 {t2['brier_mean']:.3f}。更换扩展队列树模型的训练轮数也会影响准确率与损失的取舍。固定 500 轮的早期对照在扩展真实缺测子集的准确率为 72.22%，与同预算 LimiX-2M 的 72.29% 接近。故扩展结果宜结合概率损失与基线设置一起判断。

[[FIG5]]

### 3.5 上下文抽样稳定性

固定首个划分的测试样品，原始上下文和两次复抽的准确率标准差如下。各预算的波动条件为同一测试集与同一带标签样品数量，三次取样结果可用于观察参考集构成的影响。

表3  同一测试集上三次上下文抽样的准确率标准差

|测量面板|标注数|LimiX-2M（百分点）|LightGBM（百分点）|
|---|---:|---:|---:|
"""
    for panel in PANELS:
        for n in BUDGETS:
            sds = []
            for model in (LM, GB):
                repeats = [row["accuracy"] for row in evidence["raw_context_repeat"]
                           if row["panel"] == panel and row["budget"] == n and row["model"] == model]
                assert len(repeats) == 2
                source = evidence["first_split_original"]
                first = unique(source, panel=panel, budget=n, model=model)["accuracy"]
                sds.append(pct(stdev([first, *repeats])))
            base += f"|{panel}|{n:,}|{sds[0]}|{sds[1]}|\n"
    base += f"""
### 3.6 Sr–Nd–Pb 同位素增量试验

D36_74 加入 Sr–Nd 后，LimiX-2M 准确率提高 {pct(srnd['accuracy'] - l2['accuracy_mean'])} 个百分点，五次划分中 {isotopes['srnd_gain_splits']} 次提高。加入 Pb 实测值后，准确率为 {pct(pb_values['accuracy'])}%，加入 Pb 可用性标记时为 {pct(pb_flags['accuracy'])}%（表4，图6）。有可用 Pb 的测试样品每次为 {pb_subset[0]['n_min']:,}–{pb_subset[0]['n_max']:,} 件。对这批样品，LimiX-2M 使用 Sr–Nd 时准确率为 {pct(pb_subset[0]['accuracy'])}%，加入 Pb 实测值后为 {pct(pb_subset[2]['accuracy'])}%，加入标记时为 {pct(pb_subset[1]['accuracy'])}%。同预算的 LightGBM 在含 Pb 实测值的方案中达 {pct(iso('D36_74+SrNd+Pb', GB)['accuracy'])}%。

表4  主队列同位素输入对照及五次划分均值

|输入|LimiX-2M 准确率（%）|LimiX-2M 宏平均 F1|LimiX-2M 对数损失|LightGBM 准确率（%）|
|---|---:|---:|---:|---:|
"""
    for name, label in ISO_PANELS.items():
        lrow, trow = (iso(name, m) for m in (LM, GB))
        base += (f"|{label}|{pct(lrow['accuracy'])}|{lrow['macro_f1_9_labels']:.3f}|"
                 f"{lrow['log_loss']:.3f}|{pct(trow['accuracy'])}|\n")
    base += f"""
[[FIG6]]

保留 Sr–Nd–Pb 实测值并复用元素遮挡后，LimiX-2M 在实测组合遮挡下的准确率由 {pct(mask_l[2]['accuracy'])}% 变为 {pct(iso('D36_74+SrNd+Pb', scenario='实测组合遮挡')['accuracy'])}%，等量随机遮挡下由 {pct(mask_l[1]['accuracy'])}% 变为 {pct(iso('D36_74+SrNd+Pb', scenario='随机遮挡')['accuracy'])}%。同预算 LightGBM 的相应准确率由 {pct(mask_t[2]['accuracy'])}% 变为 {pct(iso('D36_74+SrNd+Pb', GB, '实测组合遮挡')['accuracy'])}%，由 {pct(mask_t[1]['accuracy'])}% 变为 {pct(iso('D36_74+SrNd+Pb', GB, '随机遮挡')['accuracy'])}%。

"""
    base += """
## 4 讨论

### 4.1 测量覆盖与地球化学信息

同一批样品上 D36_74 的平均表现较好，表明扩展测量及其衍生变量可增加判别信息。D36_74 既引入新的实测浓度，也引入由这些测量构造的地球化学比值与异常指标。本文的面板比较估计的是这些变量的组合贡献。主量参数可表征结晶分异等造成的全岩变化，Nb、Th 和重稀土等元素比值也可帮助比较源区富集及弧环境信号[1,3]。在元素测量缺失情况下，多个衍生量可能同时失效。因此将侧重点放在可比面板和成组缺测，比只用缺测比例概括样品质量更贴近实际分析组合。

仅测量标记也携带预测信息，可能反映构造类别与文献来源、地区、分析设备或研究目标间的相关性。按训练侧真实组合遮挡的设计保留了测量项共现结构，同时将测量组合独立施加到本来齐全的测试样品。原始真实缺测样品还受原始选样与测量决策影响，遮挡试验与真实缺测试验从不同条件估计模型受测量覆盖变化的程度。

### 4.2 类别重叠及地质复核

弧后扩张中心受不同程度的地幔熔融与俯冲带物质影响，出现近似洋中脊或弧类的化学组合在地质上是可理解的[1,3–4]。图1所示比值分布与图4中的类别交叠相容。大陆裂谷和大洋高原类别内的空间差异、源区混合以及样品代表性也可能影响召回率。当前证据来自汇编标签和全岩整体分布，尚缺逐样品的岩相和构造资料核验，误判的具体成因有待从剖面产状、时空关系及原始文献逐例确认。

### 4.3 标注预算与使用边界

在相同标注预算内，LimiX-2M 的判别表现可用于安排待复核样品的顺序。全量树模型展示了充分标注时可以达到的库内水平。扩展队列中仅用完整样品作为参考的对照表明，当测试对象的测量结构与参考集明显不同，扩大参考集测量组合同样重要。实际判别宜保留参考样品类别、测量覆盖和预测概率三个信息，并用来源明确的区域样品检查主要混淆。

库内随机切分可能让相近地质单元、分析批次或资料源跨训练和测试。主队列与扩展队列共享资料汇编，当前结果的有效范围为已有构造类别、冻结样品筛选及库内数据条件。以后若对未见的地质单元开展应用，需要先核对实体重复和构造标签，并通过地质单元或原文献分组检验。

### 4.4 同位素信息与分析覆盖

Sr–Nd 对 LimiX-2M 的小幅增益与同位素可记录源区差异的认识相容。Pb 可用性标记的得分较高，显示分析覆盖本身与构造标签有关。分析项目可能随研究对象、来源和采样区域而变，这种关联需要结合资料来源解读。Pb 覆盖约为主队列的四分之一，实测比值未经统一年龄校正。在固定 2,048 件预算下，Pb 实测值给两种模型带来的变化方向不同。由此得到的是特定样品构成与模型设置下的预测表现。用 Pb 比值解释源区演化还需要核查年龄、分析来源与区域地质。

## 5 结论

（1）在 24,251 件玄武岩主队列中，LimiX-2M 在有限标注下有较好的九类构造判别表现。2,048 件、D36_74 时准确率为 {ACC2}%。全量 LightGBM 的准确率为 {FULL}%。

（2）扩展测量面板、参考集缺测组成及测试样品的遮挡方式均影响结果。仅测量标记也携带一定的标签关联信息，需要结合浓度和比值证据理解。

（3）在 33,293 件真实部分缺测的扩展样品中，混合上下文下 LimiX-2M 达到 {NAT}% 的准确率和 {LOSS} 的对数损失。弧后盆地等类别仍宜保留地质复核环节。本文的效果估计适用于库内随机验证。

（4）同预算下加入 Sr–Nd 后，LimiX-2M 准确率为 {SRND}%。继续加入 Pb 实测比值后为 {PB}%，实测组合遮挡与等量随机遮挡之间仍存在明显差距。

## 数据及代码说明

实验冻结入口为 src/run_d36_srnd_partial_chemistry_2026_11_13.py 与 src/run_limix2m_missing_d36.py。主实验、扩展统计、审计与图件脚本分别为 src/run_limix2m_geology_paper.py、src/run_limix2m_geology_expansion.py、src/audit_limix2m_geology_paper.py 和 src/plot_limix2m_geology_paper.py。结果目录 reports/modeling/limix2m_geology_paper_2026-11-17 保存划分协议、分类型结果、预测概率和审计校验和。原始汇编及再分发遵循各来源资料的条款。

同位素对照由 src/run_limix2m_geology_isotopes.py 复现，图6 由 src/plot_limix2m_geology_isotopes.py 生成。结果目录 reports/modeling/limix2m_geology_isotopes_2026-11-17 保存逐样品概率、各次划分的配对差值与审计记录。

## 参考文献

[1] Pearce J A. Geochemical fingerprinting of oceanic basalts with applications to ophiolite classification and the search for Archean oceanic crust. Lithos, 2008, 100, 14–48. https://doi.org/10.1016/j.lithos.2007.06.016

[2] Vermeesch P. Tectonic discrimination diagrams revisited. Geochemistry, Geophysics, Geosystems, 2006, 7, Q06017. https://doi.org/10.1029/2005GC001092

[3] Ueki K, Hino H, Kuwatani T. Geochemical discrimination and characteristics of magmatic tectonic settings: A machine-learning-based approach. Geochemistry, Geophysics, Geosystems, 2018, 19, 1327–1347. https://doi.org/10.1029/2017GC007401

[4] Doucet L S, Tetley M G, Li Z X, et al. Geochemical fingerprinting of continental and oceanic basalts: A machine learning approach. Earth-Science Reviews, 2022, 233, 104192. https://doi.org/10.1016/j.earscirev.2022.104192

[5] Wang Y, Zhang X, Yu H, et al. LimiX-2M: Mitigating low-rank collapse and attention bottlenecks in tabular foundation models. Proceedings of Machine Learning Research, 2026, 306, 132541–132570. https://proceedings.mlr.press/v306/wang26kw.html

[6] Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 2017, 30. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
""".format(ACC2=pct(l2["accuracy_mean"]), FULL=pct(full["accuracy_mean"]),
           NAT=pct(nat_l["accuracy"]), LOSS=f"{nat_l['log_loss']:.3f}",
           SRND=pct(srnd["accuracy"]), PB=pct(pb_values["accuracy"]))
    return base


def font(style, name="Microsoft YaHei", size=11, bold=False, color="21333D"):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string(color)
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def make_docx(article, path, figure_paths=None, title=TITLE, subtitle=SUBTITLE):
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(1.9), Cm(1.9)
    section.left_margin, section.right_margin = Cm(2.0), Cm(2.0)
    for style, size, color, bold in [
        ("Normal", 10.5, "21333D", False), ("Title", 20, "163C51", True),
        ("Subtitle", 13, "4E6574", False), ("Heading 1", 15, "163C51", True),
        ("Heading 2", 12, "1E6480", True), ("Caption", 9, "3E515D", False),
    ]:
        font(document.styles[style], size=size, color=color, bold=bold)
    normal = document.styles["Normal"].paragraph_format
    normal.line_spacing = 1.21
    normal.space_after = Pt(6)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    table = None
    for line in article.splitlines():
        if not line.strip():
            table = None
        elif line.startswith("|"):
            vals = line.strip("|").split("|")
            if all(not v.strip(":-") for v in vals):
                continue
            if table is None:
                table = document.add_table(rows=0, cols=len(vals), style="Light Shading Accent 1")
                head = True
            else:
                head = False
            row = table.add_row()
            properties = row._tr.get_or_add_trPr()
            properties.append(OxmlElement("w:cantSplit"))
            if head:
                properties.append(OxmlElement("w:tblHeader"))
            for cell, value in zip(row.cells, vals):
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.05
                run = paragraph.add_run(value)
                run.bold, run.font.size = head, Pt(8.5)
        elif line.startswith("[[FIG"):
            index = int(line[5]) - 1
            if figure_paths is None:
                figure, label = FIGURES[index]
                folder = ISOTOPE_RESULTS / "figures_zh" if figure.startswith("图6") else FIG_DIR
                image = folder / (figure + ".png")
            else:
                image, label = figure_paths[index]
            p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(image), width=Cm(16.5))
            document.add_paragraph(label, "Caption")
        elif line.startswith("# "):
            document.add_paragraph(line[2:], "Title")
        elif line.startswith("### "):
            document.add_heading(line[4:], level=2)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line == subtitle:
            document.add_paragraph(line, "Subtitle")
        elif line.startswith("**"):
            _, prefix, body = line.split("**", 2)
            p = document.add_paragraph()
            p.add_run(prefix).bold = True
            p.add_run(body)
        else:
            p = document.add_paragraph(line)
            if line.startswith(("表1 ", "表2 ", "表3 ", "表4 ", "表5 ", "表6 ", "表7 ")):
                p.paragraph_format.keep_with_next = True
                p.runs[0].bold = True
            if len(line) > 3 and line[0] == "[" and line[2] == "]":
                p.paragraph_format.keep_together = True
                p.runs[0].font.size = Pt(9)
    for table in document.tables:
        rows = list(table.rows)
        linked = rows[:-1] if len(rows) <= 7 else rows[:1]
        for row in linked:
            for cell in row.cells:
                cell.paragraphs[-1].paragraph_format.keep_with_next = True

    document.core_properties.title = title
    document.core_properties.subject = subtitle
    document.core_properties.author = ""
    document.save(path)


def main():
    evidence = json.loads((RESULTS / "manuscript_evidence.json").read_text(encoding="utf-8"))
    isotopes = json.loads((ISOTOPE_RESULTS / "manuscript_evidence.json").read_text(encoding="utf-8"))
    article = make_article(evidence, isotopes)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    make_docx(article, DELIVERY / "玄武岩构造环境判别_LimiX-2M_论文初稿_同位素修订.docx")


if __name__ == "__main__":
    main()
