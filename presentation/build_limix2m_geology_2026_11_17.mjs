import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const [workspaceDir, skill, runtimePython, revision = ""] = process.argv.slice(2);
const stage = path.join(workspaceDir, ".codex_tmp", "limix2m_geology_ppt_classification");
const results = path.join(workspaceDir, "reports", "modeling", "limix2m_geology_paper_2026-11-17");
const isotopeResults = path.join(path.dirname(results), "limix2m_geology_isotopes_2026-11-17");
const central = path.join(path.dirname(results), "limix2m_panel_control_2026-11-17");
const baselines = path.join(path.dirname(results), "limix2m_geology_baselines_2026-11-17");
const spider = path.join(path.dirname(results), "tectonic_spider_2026-11-14");
const classification = path.join(workspaceDir,"figures","current_basalt_classification_2026-11-18");
const delivery = path.join(workspaceDir, "delivery", "limix2m_玄武岩_2026-11-17");
const filename = `元素测量组合对玄武岩构造环境判别的影响_研究汇报${revision}.pptx`;
const data = JSON.parse(await fs.readFile(path.join(results, "manuscript_evidence.json"), "utf8"));
const isotopes = JSON.parse(await fs.readFile(path.join(isotopeResults, "manuscript_evidence.json"), "utf8"));
const benchmark = JSON.parse(await fs.readFile(path.join(baselines, "summary.json"), "utf8"))
  .sort((a,b) => b.accuracy-a.accuracy);
const maskFile = (await fs.readFile(path.join(central, "paired_mask_scores.csv"), "utf8")).trim().split(/\r?\n/);
const restricted = (model) => maskFile.slice(1).map(row => row.split(","))
  .filter(row => row[1] === "扩展14项随机遮挡" && row[2] === model)
  .reduce((sum, row) => sum + Number(row[4])/5, 0);

const presentation = Presentation.create({slideSize: {width: 1280, height: 720}});
const font = "Microsoft YaHei";
const colors = {
  ink: "#173746", navy: "#123641", teal: "#1E6480", rust: "#9D5637",
  light: "#D6E1E4", slate: "#4C626B", white: "#FFFFFF",
};
const line = {style: "solid", fill: "none", width: 0};
const models = ["LimiX-2M", "LightGBM"];
const percent = (n, digits = 2) => (100 * n).toFixed(digits);
const fixed = (n) => n.toFixed(3);
const each = (rows, fields) => {
  const filtered = rows.filter(row => Object.entries(fields).every(([key, value]) => row[key] === value));
  if (filtered.length !== 1) throw new Error(`Expected a unique evidence row for ${JSON.stringify(fields)}`);
  return filtered[0];
};
const main = (model, budget=2048, panel="D36_74") => each(data.main, {model, budget, panel});
const mask = (model, scenario) => each(data.mean, {model, scenario, panel:"D36_74", budget:2048, subset:"全部"});
const expand = (model, subset="D36部分缺测", context="mixed") => each(data.expansion, {model, subset, context});
const category = label => each(data.class, {model:models[0], class:label});
const [limix, tree] = models.map(m => main(m));
const full = main("全量LightGBM", 19400);
const iso = (panel, subset="全部", scenario="原始测试", model=models[0]) =>
  each(isotopes.summary, {panel, subset, scenario, model});
const srnd = "D36_74+SrNd";
const pb = "D36_74+SrNd+Pb";
const pbFlags = "D36_74+SrNd+Pb_flags";
const source = "项目冻结数据与试验汇总 · reports/modeling/limix2m_geology_paper_2026-11-17/";
const isoSource = "同位素对照与逐样品结果 · reports/modeling/limix2m_geology_isotopes_2026-11-17/";
const paper = {
  pearce: "Pearce (2008), Lithos. https://doi.org/10.1016/j.lithos.2007.06.016",
  ueki: "Ueki et al. (2018), G-Cubed. https://doi.org/10.1029/2017GC007401",
  doucet: "Doucet et al. (2022), Earth-Science Reviews. https://doi.org/10.1016/j.earscirev.2022.104192",
  limix: "Wang et al. (2026), ICML, LimiX-2M. https://proceedings.mlr.press/v306/wang26kw.html",
  tree: "Ke et al. (2017), NeurIPS, LightGBM. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
};

function box(s, content, x, y, width, height, options = {}) {
  const {size=24, color=colors.ink, bold=false, align="left", name=content.slice(0, 24)} = options;
  const shape = s.shapes.add({
    name, geometry:"textbox", position:{left:x, top:y, width, height}, fill:"none", line,
  });
  shape.text = content;
  shape.text.style = {
    typeface:font, fontSize:size, color, bold, alignment:align, verticalAlignment:"top",
    wrap:"square", autoFit:"none", insets:0,
  };
  return shape;
}

function rule(s, x, y, width, color=colors.light) {
  s.shapes.add({
    name:"分隔线", geometry:"rect", position:{left:x, top:y, width, height:2}, fill:color, line,
  });
}

function newSlide(title, note, sources=[]) {
  const slide = presentation.slides.add();
  slide.background.fill = colors.white;
  box(slide, title, 65, 50, 1150, 68, {size:43, bold:true});
  rule(slide, 65, 120, 1150);
  slide.speakerNotes.text = [note, source, ...sources].join("\n\n");
  return slide;
}

function footer(slide, context) {
  rule(slide, 65, 650, 1150);
  box(slide, context, 65, 663, 1080, 25, {size:16, color:colors.slate});
  box(slide, String(presentation.slides.items.length).padStart(2, "0"), 1158, 663, 55, 25,
    {size:16, color:colors.slate, align:"right"});
}

async function figure(slide, name, y=151, height=465) {
  const folder = name.startsWith("图6") ? path.join(isotopeResults, "figures_zh")
    : ["训练曲线", "固定基础测项的遮挡对照"].includes(name) ? central
    : name.endsWith("分类图") ? classification
    : name.startsWith("nine_tectonic") ? spider : path.join(results, "figures_zh");
  const png = await fs.readFile(path.join(folder, `${name}.png`));
  slide.images.add({
    name, alt:name.replace("_", " "),
    blob:png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength),
    contentType:"image/png", fit:"contain", position:{left:51, top:y, width:855, height},
  });
}

function side(slide, lines, y=168) {
  rule(slide, 931, y, 56, colors.teal);
  let top = y + 18;
  for (const [em, desc, height=97] of lines) {
    box(slide, em, 931, top, 280, 39, {size:27, color:colors.teal, bold:true});
    box(slide, desc, 931, top + 40, 272, height, {size:22, color:colors.ink});
    top += height + 54;
  }
}

{
  const s = presentation.slides.add();
  s.background.fill = colors.navy;
  rule(s, 84, 145, 89, "#8EC2CF");
  box(s, "元素测量组合对玄武岩\n构造环境判别的影响", 81, 181, 1100, 215,
    {size:76, color:colors.white, bold:true});
  box(s, "九类构造环境样品的配对检验", 86, 446, 1100, 61,
    {size:35, color:"#C7DEE3"});
  box(s, "九类构造环境 · 中文论文初稿配套汇报", 86, 618, 1050, 35, {size:21, color:"#C7DEE3"});
  s.speakerNotes.text = [source, paper.limix, paper.doucet].join("\n\n");
}

{
  const s = newSlide("九类玄武岩的标准化元素谱",
    "各谱点为可用实测值的中位数。Pb 表示元素浓度。主数据集 GEOROC 20,211 件，EarthChem／PetDB 3,724 件，其他 316 件。",
    [paper.pearce, paper.ueki, paper.doucet, "Sun & McDonough (1989) https://doi.org/10.1144/GSL.SP.1989.042.01.19",
     "https://georoc.eu/ · https://earthchem.org/petdb"]);
  await figure(s, "nine_tectonic_spider_template_1989_cs_u_pb");
  side(s, [
    ["24,251 件", "主队列均测有共同 22 项，覆盖九类构造环境。", 86],
    ["7,956 件", "至少缺少一项 D36 测量。", 68],
    ["元素谱形", "大陆弧、岛弧与洋内弧具有 Nb–Ta 低谷。", 96],
  ]);
  footer(s, "图1｜原始地幔标准化多元素中位谱");
}

{
  const s = newSlide("TAS 岩石分类与构造类别分布",
    "采用当前主数据集 24,251 件样品，主量氧化物总量归一至 100%。图示范围 SiO2 40%–60%、总碱 0%–12%；图外样品保留在完整数据和实验中。菱形为各类分量中位数。岩石化学字段与来源构造标签分别表述。",
    ["Le Bas et al. (1986) https://doi.org/10.1093/petrology/27.3.745",
     "图件及完整样品源表 · figures/current_basalt_classification_2026-11-18/"]);
  await figure(s,"TAS分类图",143,476);
  side(s, [
    ["24,251 件", "与本文主数据集相同。", 66],
    ["主量归一化", "SiO2 与总碱反映岩石化学分类位置。", 91],
    ["构造类别重叠", "不同构造类可落入相同岩石字段。", 88],
  ]);
  footer(s,"图2｜TAS 岩石分类，颜色为九类构造环境，菱形为各类中位数");
}

{
  const s = newSlide("九类玄武岩的 AFM 系列分布",
    "24,251 件主数据集样品。A 为 Na2O＋K2O，F 为 Fe2O3(T)，M 为 MgO。全铁由 FeO(T) 除以 0.8998 换算，分界同时转换到 Fe2O3(T) 基准。分界据 Irvine–Baragar (1971)，数值实现参考 Rickwood (1989)。",
    ["https://doi.org/10.1139/e71-055","https://doi.org/10.1016/0024-4937(89)90028-5"]);
  await figure(s,"AFM分类图",137,494);
  side(s, [
    ["相同铁基准", "样品与分界均以 Fe2O3(T) 表示全铁。", 90],
    ["逐类展示", "图中标出各构造类样品数，便于比较系列分布。", 102],
  ],193);
  footer(s,"图3｜AFM 系列图，虚线为 Irvine–Baragar 分界");
}

{
  const s = newSlide("测量面板与标注样品设计",
    "两模型和两面板使用同一上下文、同一测试集。按类别抽取嵌套预算，元素及同位素真实缺测保留。LimiX-2M 四成员推理，模型权重冻结。LightGBM 在预算内早停后重拟合。同位素对照固定为 2,048 件标注。",
    [paper.limix, paper.tree, isoSource]);
  const groups = [
    {x:65, w:545, title:"测量面板", labels:[
      ["D22_42", "22 项共同测量及其衍生变量"],
      ["D36_74", "36 项测量及比值、异常等衍生变量"],
      ["Sr–Nd–Pb 同位素", "加实测比值，并设置 Pb 可用性标记对照"],
    ]},
    {x:666, w:545, title:"样品划分与模型比较", labels:[
      ["五次随机 80∶20", "每个划分两模型共用 4,851 件测试样品"],
      ["512 ／ 2,048 ／ 4,096", "两种模型使用相同的带标签样品"],
      ["19,400 件全量参照", "LightGBM 使用每次划分的全训练集"],
    ]},
  ];
  for (const a of groups) {
    box(s, a.title, a.x, 157, a.w, 50, {size:33, bold:true, color:colors.teal});
    for (const [index, [heading, detail]] of a.labels.entries()) {
      const y = 235 + index * 122;
      rule(s, a.x, y - 10, a.w);
      box(s, heading, a.x, y + 3, a.w, 45, {size:29, bold:true});
      box(s, detail, a.x, y + 53, a.w, 53, {size:24, color:colors.slate});
    }
  }
  footer(s, "主队列每次 19,400 件训练、4,851 件测试｜五次分层重划");
}

{
  const s = newSlide("LightGBM 拟合与验证过程",
    "显示选轮阶段实测日志。首次划分 1,638 件用于内部拟合，410 件用于内部验证。按验证损失选择轮数，再用全部 2,048 件重新拟合。外层测试集独立评分。",
    [paper.limix, paper.tree]);
  await figure(s, "训练曲线");
  side(s, [
    [`${percent(limix.accuracy_mean)}% / ${percent(tree.accuracy_mean)}%`, "LimiX-2M 和 LightGBM\n在 D36_74、2,048 件标注下的准确率。", 104],
    ["冻结权重", "LimiX-2M 在预测时读取参考样品。", 65],
    [`${percent(full.accuracy_mean)}%`, "全量 LightGBM 准确率，使用 19,400 件标注。", 85],
  ]);
  footer(s, "图4｜首次划分的内部训练日志；右侧准确率来自外层测试");
}

{
  const s = newSlide("相同训练样品下的模型比较",
    "各划分复用完全一致的 2,048 件带标签样品与 4,851 件测试样品。XGBoost 与 CatBoost 由内部验证损失选轮；随机森林使用训练侧中位数填补和缺失标记。五模型输入为相同的扩展元素面板。",
    ["基线结果 · reports/modeling/limix2m_geology_baselines_2026-11-17/scores_summary.csv"]);
  const columns = [65, 492, 765, 995];
  for (const [j, heading] of ["模型", "准确率（%）", "宏平均 F1", "对数损失"].entries())
    box(s, heading, columns[j], 164, 240, 46, {size:27, bold:true});
  for (const [i, r] of benchmark.entries()) {
    const y = 229+i*69;
    rule(s,65,y-13,1150);
    const values = [r.model, percent(r.accuracy), fixed(r.macro_f1_9_labels), fixed(r.log_loss)];
    values.forEach((v,j) => box(s,v,columns[j],y,220,44,{size:28,bold:i===0,color:i===0?colors.teal:colors.ink}));
  }
  footer(s, "五次库内划分均值｜2,048 件带标签样品，扩展元素输入");
}

{
  const s = newSlide("等量遮挡下的测量组合效应",
    "训练侧缺测组合随机分配给原本完整的测试样品，抽取时与类别独立。等量随机遮挡与实测组合遮挡对同一样品去除相同数目的实测项，随后重算输入。同位素扩展复用同批样品和遮挡位置，并保留 Sr–Nd–Pb 实测值。仅输入元素有无标记在未改动样品上得到 28.41% 准确率。",
    [paper.ueki, isoSource]);
  const random = mask(models[0], "随机遮挡");
  const empirical = mask(models[0], "实测组合遮挡");
  await figure(s, "固定基础测项的遮挡对照");
  side(s, [
    [`${percent(empirical.accuracy)}% / ${percent(restricted(models[0]))}%`, "LimiX-2M 真实组合与扩展 14 项随机遮挡", 86],
    [`${percent(mask(models[1], "实测组合遮挡").accuracy)}% / ${percent(restricted(models[1]))}%`, "LightGBM 在同一对照中的准确率", 71],
    [`${percent(random.accuracy)}%`, "全 36 项随机时的 LimiX-2M 准确率", 80],
  ]);
  footer(s, "图5｜五次划分；同一样品各组遮掉相同数量的实测项");
}

{
  const s = newSlide("九类构造环境的主要混淆",
    "在主队列 2,048 件标注、D36_74 的条件下，LimiX-2M 将弧后盆地错判为洋中脊 207 条、洋内弧 114 条。重复划分的测试记录可以包含同一实体样品。",
    [paper.pearce, paper.doucet]);
  await figure(s, "图4_九类构造判别混淆", 145, 480);
  side(s, [
    [`弧后盆地 ${percent(category("BAB").recall, 1)}%`, "多误判为洋中脊或洋内弧。", 87],
    [`大洋高原 ${percent(category("OP").recall, 1)}%`, "类别内部地球化学变化值得逐样品对照。", 87],
    [`大陆裂谷 ${percent(category("CR").recall, 1)}%`, "以上均为 LimiX-2M 召回率。", 50],
  ]);
  footer(s, "图6｜五次测试记录合并、按真实类别归一；D36_74，2,048 件标注");
}

{
  const s = newSlide("混合缺测参考样品的收益与代价",
    "扩展队列 49,676 件的随机五折。两模型输入 36 项浓度的常用对数。每折 2,048 件参考样品，混合组在各构造类内替换至多一半完整样品，保持类别频数。每个测试子集在两组中包含相同实体。完整测试样品 16,383 件，真实缺测测试样品 33,293 件。替换同时改变完整样品数量与地球化学组成。",
    [paper.limix, paper.tree]);
  const cols = [65, 370, 635, 947];
  ["测试子集", "模型", "准确率（%）", "变化（百分点）"].forEach((v,i) =>
    box(s,v,cols[i],161,[280,240,295,268][i],45,{size:27,bold:true}));
  let index = 0;
  for (const [subset,label] of [["D36齐全","完整样品"],["D36部分缺测","真实缺测样品"]]){
    for (const model of models){
      const before=expand(model,subset,"complete").accuracy;
      const after=expand(model,subset,"mixed").accuracy;
      const change=(after-before)*100;
      const values=[label, model, `${percent(before)} → ${percent(after)}`,
        `${change>0?"+":""}${change.toFixed(2)}`];
      const y=233+index*73;
      rule(s,65,y-13,1150);
      values.forEach((v,i)=>box(s,v,cols[i],y,[280,240,295,268][i],50,
        {size:29,bold:i===3,color:i===3?colors.teal:colors.ink}));
      index++;
    }
  }
  box(s,"每折参考样品数及类别频数相同，替换同时改变了所用实体样品和测量覆盖。",65,551,1140,62,
    {size:25,color:colors.slate});
  footer(s, "准确率依次为仅完整参考集、混合参考集｜相同测试样品的随机五折");
}

{
  const s = newSlide("预测概率与参考样品组成",
    "D36_74 2,048 件预算的校准值按五次划分汇总。参考集抽样波动仅在第一个测试划分内评价，所列区间为三次上下文抽样结果范围，不是置信区间。",
    [paper.limix, paper.tree]);
  const lCal = main("LimiX-2M校准");
  const tCal = main("LightGBM校准");
  const rows = [...data.first_split_original, ...data.raw_context_repeat].filter(row =>
    row.panel === "D36_74" && row.budget === 2048 && row.model === models[0]);
  const values = rows.map(row => row.accuracy);
  const range = `${percent(Math.min(...values))}%–${percent(Math.max(...values))}%`;
  const low = expand(models[0], "D36部分缺测", "complete");
  const high = expand(models[0]);
  const columns = [
    {x:65, title:"多类对数损失", stats:[
      [`${fixed(limix.log_loss_mean)} → ${fixed(lCal.log_loss_mean)}`, "LimiX-2M 温度校准前后"],
      [`${fixed(tree.log_loss_mean)} → ${fixed(tCal.log_loss_mean)}`, "LightGBM 温度校准前后"],
    ]},
    {x:666, title:"LimiX-2M 参考样品", stats:[
      [range, "主队列固定测试集，三次抽取参考样品"],
      [`${percent(low.accuracy)}% → ${percent(high.accuracy)}%`, "真实缺测子集\n参考集由仅完整样品改为混合样品"],
    ]},
  ];
  for (const a of columns) {
    box(s, a.title, a.x, 160, 545, 53, {size:34, bold:true});
    for (const [i, [metric, description]] of a.stats.entries()) {
      const y = 236 + i * 170;
      rule(s, a.x, y - 7, 545);
      box(s, metric, a.x, y + 12, 540, 60, {size:39, bold:true, color:colors.teal});
      box(s, description, a.x, y + 80, 525, 81, {size:24, color:colors.slate});
    }
  }
  footer(s, "2,048 件标注｜主队列 D36_74，扩展队列 D36 浓度对数");
}

{
  const s = newSlide("人工遮挡评估与缺测样品参与训练",
    "当前人工遮挡仅用于原本完整的测试样品，重新计算特征后预测。模型未使用人工遮挡样品训练，也未用它们扩充 LimiX-2M 上下文。混合参考集实验使用真实缺测、带标签的样品。现有结果说明两类测试对象出现取舍，保留完整参考样品并额外追加缺测样品的情形尚待检验。",
    [source]);
  box(s,"人工遮挡评估",65,166,540,53,{size:34,bold:true,color:colors.teal});
  box(s,"真实缺测参考样品",666,166,540,53,{size:34,bold:true,color:colors.teal});
  rule(s,65,230,545);
  rule(s,666,230,545);
  box(s,"改变测试样品的测量组合，检查信息损失与预测稳定性。",65,256,520,111,{size:29});
  box(s,"改变训练或上下文样品的组成，观察对完整和缺测对象的影响。",666,256,520,111,{size:29});
  box(s,"人工遮挡训练能否提高准确率，仍需直接试验。",65,407,520,115,{size:28,color:colors.slate});
  box(s,"缺测样品预测改善，同时完整样品的准确率下降。",666,407,520,115,{size:28,color:colors.slate});
  footer(s,"后续对照｜保留完整参考样品并追加缺测样品，分别评价两个测试子集");
}

{
  const s = newSlide("Sr–Nd–Pb 同位素的增量检验",
    `主队列有 ${isotopes.cohort.Sr.toLocaleString("en-US")} 件可用 Sr，${isotopes.cohort.Nd.toLocaleString("en-US")} 件可用 Nd，${isotopes.cohort.any_Pb.toLocaleString("en-US")} 件至少有一项可用 Pb。比值经质控，采用实测值，缺测保留。每次划分的两模型和四种输入共用带标签样品及测试集。Pb 标记反映测量覆盖与构造标签的关系。`,
    [paper.ueki, isoSource]);
  await figure(s, "图6_同位素输入对照");
  const sr = iso(srnd), pv = iso(pb), pf = iso(pbFlags);
  const srOnPb = iso(srnd, "至少一项Pb"), pvOnPb = iso(pb, "至少一项Pb");
  side(s, [
    [`${percent(sr.accuracy)}% → ${percent(pv.accuracy)}%`, "从加 Sr–Nd 到再加 Pb 实测值的总体准确率", 89],
    [`${percent(pf.accuracy)}%`, "在 Sr–Nd 基础上加 Pb 可用性标记的准确率", 82],
    [`${percent(srOnPb.accuracy)}% / ${percent(pvOnPb.accuracy)}%`, "有实测 Pb 的测试子集\n加入 Pb 比值前后", 82],
  ]);
  footer(s, "图7｜相同的 2,048 件带标签样品；右侧为 LimiX-2M 的五次划分均值");
}

{
  const s = newSlide("适用条件与地质复核",
    "当前范围是已有类别和资料汇编内的随机验证。主队列和扩展队列可能包含同一实体，地质相关样品和来源可跨集合。限定为筛查应用并记录测量覆盖。判断成因需要区域地质、样品背景与原始分析证据。",
    [paper.pearce, paper.doucet]);
  const summary = [
    ["有限标注", "同预算下，LimiX-2M 在主队列上取得更高准确率。充分标注时，全量 LightGBM 的准确率更高。"],
    ["测量覆盖", "共同 22 项与缺测数相同时，真实测量组合的判别表现优于扩展项内随机组合。"],
    ["同位素比值", "加入 Sr–Nd 有小幅增益。加入 Pb 实测值后，LimiX-2M 整体准确率回落。"],
    ["地质判定", "将预测用于挑选待复核样品。弧后盆地等混淆类别须结合产状、区域背景与微量元素证据。"],
  ];
  for (const [i, [heading, detail]] of summary.entries()) {
    const y = 162 + i * 116;
    box(s, heading, 65, y, 240, 49, {size:34, color:colors.teal, bold:true});
    box(s, detail, 310, y + 2, 883, 89, {size:27});
    if (i < summary.length - 1) rule(s, 65, y + 89, 1150);
  }
  footer(s, "证据范围｜现有资料库内的随机划分");
}

await fs.mkdir(stage, {recursive:true});
await fs.mkdir(delivery, {recursive:true});
const candidate = path.join(stage, `candidate${revision}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill, "container_tools", "artifact_tool_utils.mjs")).href);
const result = await finalizePresentation({
  workspaceDir,
  candidatePath:candidate,
  finalPath:path.join(delivery, filename),
  pythonExecutable:runtimePython,
  integrityValidatorPath:path.join(skill, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skill, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-bullet-geometry","--validate-heading-fit"],
  requiredNativeTableOwnerSlides:[],
  requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:"design", families:[font]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(stage, `validation${revision}.json`),
});
console.log(result);

const render = path.join(stage, `render${revision}`);
await fs.mkdir(render, {recursive:true});
for (const [i, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({slide, format:"png", scale:1});
  await fs.writeFile(path.join(render, `slide-${String(i + 1).padStart(2,"0")}.png`),
    new Uint8Array(await png.arrayBuffer()));
}
