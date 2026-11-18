"""Apply the geological manuscript review to the existing experiment draft."""

import csv
import json
import re
from statistics import mean

from write_limix2m_geology_paper import (
    DELIVERY, FIG_DIR, FIGURES, ISOTOPE_RESULTS, RESULTS, make_article, make_docx, pct,
)


CONTROL = RESULTS.with_name("limix2m_panel_control_2026-11-17")
SPIDER = RESULTS.with_name("tectonic_spider_2026-11-14")
CLASSIFICATION = RESULTS.parents[2] / "figures/current_basalt_classification_2026-11-18"
BASELINES = RESULTS.with_name("limix2m_geology_baselines_2026-11-17")
TITLE = "元素测量组合对玄武岩构造环境判别的影响"
SUBTITLE = "九类构造环境样品的配对检验"


def revise(text, scores, history, baselines):
    def paragraph(prefix, value):
        nonlocal text
        blocks = text.split("\n\n")
        index, = (i for i, block in enumerate(blocks) if block.strip().startswith(prefix))
        blocks[index] = value
        text = "\n\n".join(blocks)

    def acc(model, scenario):
        return mean(float(r["accuracy"]) for r in scores if r["model"] == model and r["scenario"] == scenario)

    lm, gb = "LimiX-2M", "LightGBM"
    real, random = "实测组合遮挡", "扩展14项随机遮挡"
    lm_gap = pct(acc(lm, real) - acc(lm, random))
    gb_gap = pct(acc(gb, real) - acc(gb, random))
    selected = sorted({int(r["selected_round"]) for r in history})
    first = min(r["seed"] for r in history)
    rounds, = {int(r["selected_round"]) for r in history if r["seed"] == first}

    paragraph("# 基于", "# " + TITLE)
    paragraph("元素测量覆盖、标注样品数量", SUBTITLE)
    paragraph("**目的**", "**目的** 玄武岩汇编中，分析项目常按组合缺失。本文检验保留共同元素测量时，缺失项目的组合如何影响九类构造环境的判别。")
    paragraph("**方法**", "**方法** 主数据集包含 24,251 件具有 22 项共同测量的玄武岩，其中 16,295 件测齐另外 14 项扩展元素。在同一批完整样品上固定共同 22 项，比较真实缺测组合和扩展项内等量随机缺测。采用 LimiX-2M 与 LightGBM 作配对比较，以 49,676 件样品组成的扩展数据集检验真实缺测，并比较 Sr–Nd–Pb 实测值与 Pb 测量标记的增量信息。")
    paragraph("**结果**", f"**结果** 在共同测量和缺测数均相同的条件下，真实组合遮挡比扩展项内随机遮挡的准确率高 {lm_gap} 个百分点（LimiX-2M）和 {gb_gap} 个百分点（LightGBM）。使用 2,048 件带标签样品时，LimiX-2M 加 Sr–Nd 后由 78.96% 升至 79.43%，再加 Pb 实测值为 79.01%。弧后盆地漏判主要指向洋中脊与洋内弧。")
    paragraph("**结论**", "**结论** 在本汇编的库内验证中，测量项目数量相同时，元素组合仍会改变判别表现。评估资料适用性时，应核对实际元素组合、同位素覆盖及易混淆的构造类别。")

    paragraph("玄武岩的主量", """玄武岩的元素组成记录了源区性质、部分熔融及岩浆演化。经典判别图据此识别洋中脊、弧及板内玄武岩，Pearce（2008）进一步用 Nb/Yb 和 Th/Yb 讨论地幔富集与俯冲组分输入[1]。

### 1.1 玄武岩构造判别与缺测资料的研究进展

Vermeesch（2006）重新估计了经典判别图的统计边界，并用分类树联合元素与同位素，通过替代分裂处理待判样品的部分缺测[2,7]。Ueki 等（2018）在八类火山岩数据上进行多变量判别与特征选择，估计不同构造环境的代表性玄武岩组成[3]。Doucet 等（2022）将机器学习判别扩展到全球大陆与大洋玄武岩，并结合地质案例检验分类结果[4]。Lei 等（2025）以 GEOROC 不完整记录为基础，通过数据增强生成样品并训练卷积神经网络[8]。

### 1.2 研究问题""")
    paragraph("在实际资料中", "同样缺测数可以对应不同的元素组合，测项可用性也可能与研究区域、分析目的和构造标签有关。本文保留共同测量，在同一批样品上比较扩展元素的真实缺测组合与等量随机缺测，量化测项组合对判别的影响。")
    paragraph("LimiX-2M 根据", "本文用可在推理时读取参考样品的 LimiX-2M [5] 与梯度提升树 LightGBM [6] 检查上述影响在两种方法上是否一致。结合九类玄武岩的元素谱、主要误判和 Sr–Nd–Pb 增量试验，讨论部分测量资料可用于哪些判别情形。")
    paragraph("主队列取自", """主数据集包括 GEOROC（https://georoc.eu/，20,211 件）及 EarthChem／PetDB（https://earthchem.org/petdb，3,724 件）。另有 Zenodo 弧玄武岩和全球汇编 216 件、Mendeley Data 55 件和 PANGAEA 45 件，入口分别为 https://zenodo.org/、https://data.mendeley.com/ 和 https://www.pangaea.de/。按实体样品标识对齐化学记录和构造标签，隔离身份或重复记录待核查的样品。清单保留逐样品文献与来源编号。

主数据集共 24,251 件，均有 22 项共同测量。其中 16,295 件测齐 36 项，7,956 件缺少至少一项扩展测量。九类为弧后盆地（BAB）、大陆弧（CA）、大陆溢流玄武岩（CFB）、大陆裂谷（CR）、岛弧（IA）、洋内弧（IOA）、洋中脊（MOR）、大洋岛（OI）和大洋高原（OP）。各类元素谱逐元素使用可用实测值，按原始地幔参考值归一化[9]。回查补入的 Cs、U 和 Pb 元素浓度用于图1的描述性比较。""")
    text = text.replace("主队列沿用随机种子 20261113–20261117 的五次分层", "主队列采用五次分层随机")
    text = text.replace("D22_42 包括", "在主数据集上构建两套输入变量。基础面板（D22_42）包括")
    text = text.replace("D36_74 在该基础", "扩展面板（D36_74）在该基础")
    text = text.replace("使用 A、B 级", "使用通过项目质控的")
    text = text.replace("缺测或未通过质控的比值保留为 NaN。", "质控检查样品身份、分析材料、数值范围及重复测量一致性。A级另有可核对的分析方法、实验室和来源记录，B级满足上述基础检查，但部分分析元数据不全。两级均参与实验，缺测或未通过检查的比值保留为 NaN。")
    paragraph("对主队列每个测试集", """既有工作已针对地球化学缺测使用替代分裂和数据增强[7–8]。本试验从训练侧缺测样品抽取测量组合，将其施加于本来测齐的测试样品，并以同一样品的完整测值为对照。

每件样品在三个遮挡组均缺少相同数量的实测项。真实组合与扩展项内随机组均固定共同 22 项，后者仅在额外 14 项中置换缺测位置。全 36 项随机组允许遮去共同测量，用作压力测试。随后重算衍生变量，模型参考样品维持相同的 2,048 件。另以 36 个元素实测标记训练 LightGBM，检查测量模式与构造标签的关联。""")
    text = text.replace("（表1，图2）", "（表1）")
    benchmark = "\n".join(
        f"|{r['model']}|{pct(r['accuracy'])}|{r['macro_f1_9_labels']:.3f}|{r['log_loss']:.3f}|"
        for r in sorted(baselines, key=lambda v: -v["accuracy"])
    )
    text = text.replace("[[FIG2]]", f"""LightGBM 的拟合与验证过程见图2。五次分割由内部验证损失选取的轮数范围为 {min(selected)}–{max(selected)}，随后用完整的训练侧参考样品重新拟合。图中展示选轮阶段的实际记录。LimiX-2M 使用冻结权重，从参考样品进行上下文推理。

[[FIG2]]

另比较三种常用树模型。XGBoost 和 CatBoost 在相同内部划分以验证损失选取迭代轮数，再用全部选定样品拟合[10–11]。随机森林使用训练侧中位数填补及缺失标记[12]。表2为相同 2,048 件带标签样品、相同扩展元素输入、同一批测试样品上的比较。对数损失由各模型的原始概率计算。

表2  扩展元素输入下五种模型的五次平均表现

|模型|准确率（%）|宏平均 F1|对数损失|
|---|---:|---:|---:|
{benchmark}
""")
    paragraph("完整实测、等量随机遮挡",f"""对 16,346 条完整测试记录作配对比较（图3）。共同 22 项保留时，真实组合与扩展项内随机两组的准确率分别为 {pct(acc(lm, real))}% 和 {pct(acc(lm, random))}%（LimiX-2M），以及 {pct(acc(gb, real))}% 和 {pct(acc(gb, random))}%（LightGBM）。两模型的差值分别为 {lm_gap} 和 {gb_gap} 个百分点。

全 36 项随机组中，83.48% 的记录遮去了至少一项共同测量，62.20% 遮去了至少一项主量氧化物。其准确率为 59.01% 和 61.82%，反映基础测量也受损的压力条件。以实际测试集的元素测量标记作为唯一输入时，LightGBM 准确率为 28.41%，多数类基线为 23.93%。""")
    text = text.replace("表2  D36_74", "表3  D36_74").replace("（表2，图4）", "（表3，图4）")
    text = text.replace("（图5）", "（表4）").replace("表3  同一测试集", "表5  同一测试集").replace("表4，图6", "表6，图5").replace("表4  主队列", "表6  主队列")
    text = text.replace("[[FIG5]]", """表4  真实缺测样品与参考样品组成的比较

|参考样品组成|LimiX-2M 准确率|LightGBM 准确率|
|---|---:|---:|
|仅使用 36 项测齐的样品|50.39%|58.73%|
|混合完整和部分缺测样品|72.29%|71.11%|""")
    text = text.replace("比值图用于", "元素谱用于")
    text = text.replace("该现象须与图1所示范围一起读。图1只绘出比值各组分实测的子集，反映类间分布重叠及同类变化，无法将逐条误判归因到单个比值。", "图1显示弧后盆地和洋中脊在部分稀土元素上的中位值接近，各谱点基于有实测值的子集。")
    text = text.replace("图1所示比值分布与图4中的类别交叠相容。", "图1中大陆弧、岛弧与洋内弧的 Nb–Ta 低谷，与通常观察到的弧类谱形一致。弧后盆地与洋中脊部分稀土元素中位值接近，也可与图4的混淆一同考察。")
    paragraph("（1）在 24,251", f"（1）当共同 22 项和缺测数固定时，按真实测量组合遮挡比在扩展元素内随机遮挡的准确率高 {lm_gap} 个百分点（LimiX-2M）和 {gb_gap} 个百分点（LightGBM）。")
    paragraph("（2）扩展测量面板", "（2）使用 2,048 件带标签样品并输入扩展元素时，LimiX-2M 准确率为 78.96%。用全部训练样品拟合的 LightGBM 为 88.56%。测量组合、参考样品构成和带标签样品数均影响判别表现。")
    expansion = json.loads((RESULTS / "manuscript_evidence.json").read_text(encoding="utf-8"))["expansion"]
    def outcome(subset, model, context):
        row, = (r for r in expansion if r["subset"] == subset and r["model"] == model and r["context"] == context)
        return row["accuracy"]

    comparison = []
    for subset, description in (("D36齐全", "完整测试样品（16,383 件）"),
                                ("D36部分缺测", "真实缺测测试样品（33,293 件）")):
        for model in (lm, gb):
            before = outcome(subset, model, "complete")
            after = outcome(subset, model, "mixed")
            comparison.append(f"|{description}|{model}|{pct(before)}|{pct(after)}|{100*(after-before):+.2f}|")
    discussion = """
### 4.5 混合参考样品对完整与缺测样品的不同影响

扩展数据集的随机五折在相同测试样品上比较两种参考集。每折均使用 2,048 件带标签样品。仅完整参考集全部测齐 36 项，混合参考集在各构造类内将至多一半完整样品替换为真实缺测样品，保持各类样品数量一致。两模型统一输入 36 项元素浓度的常用对数。表7汇总完整与缺测测试子集的配对结果。

表7  参考集组成改变后两个测试子集的准确率

|测试子集|模型|仅完整参考集（%）|混合参考集（%）|变化（百分点）|
|---|---|---:|---:|---:|
""" + "\n".join(comparison) + """

混合参考集使 LimiX-2M 对真实缺测样品的准确率提高 21.89 个百分点，同时对完整样品降低 4.39 个百分点。LightGBM 分别提高 12.38 个百分点和降低 3.88 个百分点。该现象与参考集覆盖更多实际测量模式相容，但替换也改变了所用实体样品及其地球化学分布。完整样品减少、样品组成变化和缺测模式的作用共同存在，当前对照尚不能估计各因素的独立贡献。

因此，真实缺测样品的收益需与完整样品的变化一并报告。实际应用可按待判样品的测量完整性检查模型表现。若要检验在保留全部完整参考样品的基础上追加缺测样品能否改善两类对象，需要进一步设置追加样品的对照。

### 4.6 人工遮挡评估与缺测参考样品的作用

本文的人工遮挡施加在原本完整的测试样品上，模型使用原来训练侧的参考样品。遮挡后重算浓度和衍生变量，再预测相同实体。该设计衡量测量缺失造成的预测变化，实验没有以人工遮挡样品训练模型或扩充 LimiX-2M 上下文。因此，遮挡试验的结果支持对测量信息损失的评估。

真实缺测参考集试验改变了模型接触的带标签样品。它支持在本资料库内采用混合参考集时，原本缺测样品的准确率提高。人工遮挡用于训练的数据增强是否进一步改善准确率，仍待直接检验。后续试验需在完整与缺测测试子集分别评价，并维持相同测试样品，防止总准确率掩盖两类对象间的取舍。

"""
    text = text.replace("## 5 结论", discussion + "## 5 结论")
    text = text.replace("预算指带标签的实体样品总数。", "")
    text = text.replace("标注预算", "带标签样品数").replace("等预算", "相同参考样品数")
    text = text.replace("同预算", "同样品数").replace("各预算", "各参考样品规模").replace("相同预算", "相同参考样品数")
    text = text.replace("件预算", "件带标签样品").replace("完整预算", "全部选定的训练样品")
    text += """\n[7] Vermeesch P. Tectonic discrimination of basalts with classification trees. Geochimica et Cosmochimica Acta, 2006, 70, 1839–1848. https://doi.org/10.1016/j.gca.2005.12.016

[8] Lei M, Li J, Cai W, et al. A new method using convolutional neural networks to distinguish basalt tectonic settings: breaking through the constraints of incomplete geochemical datasets. Earth Science Informatics, 2025, 18, 506. https://link.springer.com/article/10.1007/s12145-025-02008-3

[9] Sun S S, McDonough W F. Chemical and isotopic systematics of oceanic basalts: implications for mantle composition and processes. Geological Society, London, Special Publications, 1989, 42, 313–345. https://doi.org/10.1144/GSL.SP.1989.042.01.19

[10] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016, 785–794. https://doi.org/10.1145/2939672.2939785

[11] Prokhorenkova L, Gusev G, Vorobev A, et al. CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 2018, 31. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

[12] Breiman L. Random forests. Machine Learning, 2001, 45, 5–32. https://doi.org/10.1023/A:1010933404324
"""
    return text, rounds


def main():
    original = json.loads((RESULTS / "manuscript_evidence.json").read_text(encoding="utf-8"))
    isotopes = json.loads((ISOTOPE_RESULTS / "manuscript_evidence.json").read_text(encoding="utf-8"))
    with (CONTROL / "paired_mask_scores.csv").open(encoding="utf-8") as file:
        scores = list(csv.DictReader(file))
    with (CONTROL / "tree_training_history.csv").open(encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    baselines = json.loads((BASELINES / "summary.json").read_text(encoding="utf-8"))
    text, rounds = revise(make_article(original, isotopes), scores, history, baselines)
    fig5 = FIGURES[5][1].replace("图6", "图5", 1).replace("D36_74", "仅使用扩展元素的输入")
    figures = [
        (SPIDER / "nine_tectonic_spider_template_1989_cs_u_pb.png",
         "图1  九类玄武岩的原始地幔标准化多元素中位谱。归一化采用 Sun 和 McDonough（1989）[9]。各谱点基于该类对应元素的可用实测值，Pb 为元素浓度。"),
        (CONTROL / "训练曲线.png",
         f"图2  LightGBM 在扩展元素面板上的内部拟合及验证过程。a，多类对数损失。b，准确率。首次划分的 2,048 件训练侧样品分成 1,638 件内部拟合及 410 件内部验证样品。虚线为验证损失最低的第 {rounds} 轮。"),
        (CONTROL / "固定基础测项的遮挡对照.png",
         "图3  遮挡前后的配对准确率变化。真实组合与扩展14项内随机均保留共同22项。全36项随机为压力测试。每件样品在三组中遮去相同项数。圆点和菱形分别表示单次划分与五次均值。"),
        (FIG_DIR / (FIGURES[3][0] + ".png"), FIGURES[3][1].replace("2,048 件预算", "2,048 件带标签样品")),
        None,
        (ISOTOPE_RESULTS / "figures_zh" / (FIGURES[5][0] + ".png"), fig5),
    ]
    text = text.replace("图6 由", "图5 由")
    text = text.replace("同位素对照由", "固定基础测量的遮挡与训练日志由 src/run_limix2m_panel_control.py 生成。补充基线由 src/run_limix2m_geology_baselines.py 复现。同位素对照由")
    text = re.sub(r"图([2-5])(?!\d)", lambda m: f"图{int(m[1])+2}", text)
    for index, item in enumerate(figures):
        if item is not None:
            path, caption = item
            caption = re.sub(r"图([2-5])(?!\d)", lambda m: f"图{int(m[1])+2}", caption)
            figures[index] = path, caption
    with (CLASSIFICATION / "TAS显示范围统计.csv").open(encoding="utf-8") as file:
        counts = dict(csv.reader(file))
    visible = int(counts["显示范围内"])
    text = text.replace("[[FIG1]]", """[[FIG1]]

为比较各构造类样品的岩石化学分布，补充 TAS 岩石分类图和 AFM 系列图。TAS 使用十项主量氧化物总量归一至 100% 后的 SiO2 与 Na2O＋K2O，字段采用 Le Maitre 模型的实现[13]。AFM 以 Na2O＋K2O、全铁和 MgO 归一化，样品与系列分界的全铁均换算为 Fe2O3(T) 基准[14–15]。两图用于描述岩石化学组成及系列分布，九类构造标签采用来源记录。

[[FIG7]]

[[FIG8]]""")
    figures.extend([
        (CLASSIFICATION / "TAS分类图.png",
         f"图2  主数据集的 TAS 岩石分类图。全部 24,251 件样品参加计算，显示 SiO2 40%–60%、总碱 0%–12% 范围内的 {visible:,} 件。其余样品保留在数据和实验中。菱形表示各构造类分量中位数。主量元素无水归一化，字段按 Le Maitre 模型实现[13]。"),
        (CLASSIFICATION / "AFM分类图.png",
         "图3  九类构造环境样品的 AFM 三元图，共 24,251 件。A 为 Na2O＋K2O，F 为 Fe2O3(T)，M 为 MgO。全铁由 FeO(T) 除以 0.8998 换算。虚线为 Irvine–Baragar 系列分界的 Rickwood 数值实现，两者均转换到相同铁基准[14–15]。"),
    ])
    text += """
[13] Le Bas M J, Le Maitre R W, Streckeisen A, Zanettin B. A chemical classification of volcanic rocks based on the total alkali-silica diagram. Journal of Petrology, 1986, 27, 745–750. https://doi.org/10.1093/petrology/27.3.745

[14] Irvine T N, Baragar W R A. A guide to the chemical classification of the common volcanic rocks. Canadian Journal of Earth Sciences, 1971, 8, 523–548. https://doi.org/10.1139/e71-055

[15] Rickwood P C. Boundary lines within petrologic diagrams which use oxides of major and minor elements. Lithos, 1989, 22, 247–263. https://doi.org/10.1016/0024-4937(89)90028-5
"""
    make_docx(text, DELIVERY / f"{TITLE}_论文_分类图补充.docx", figures, TITLE, SUBTITLE)


if __name__ == "__main__":
    main()
