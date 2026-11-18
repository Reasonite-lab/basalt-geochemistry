# LimiX-2M 构造环境判别工具

基于 LimiX-2M 结构化数据基础模型（stable-ai，Apache-2.0）的玄武岩构造环境判别桌面软件。
模型以**上下文学习**方式工作：预测时对照 1024 条带标签的冻结参考样品（来自 expansion13 冻结数据，按类别分层抽取）。

## 运行

- 发布版适用于 Windows 10/11 64 位，无需安装 Python 或配备显卡。
- 完整解压 CPU 版 ZIP，双击 `LimiX2MPredictor_CPU.exe`。请保留同目录的 `_internal` 文件夹。
- 首次分析自动加载模型；CPU 推理耗时取决于样品数量与处理器。
- GPU 版本地构建入口为 `tools/build_local.py`，不作为 GitHub 发行包。

## 输入

- CSV、TXT 或 XLSX，每行一个未知样品；
- 元素列名见导出的空白模板；**留空表示缺测**（LimiX-2M 原生支持缺失值）；
- 最低覆盖：全部 10 项主量氧化物 + 任意 8 项微量元素（共 ≥ 18 项实测）；
- 已测量的元素必须为有限正数；一次最多 500 个样品。

## 输出

九类构造环境（BAB、CA、CFB、CR、IA、IOA、MOR、OI、OP）概率表：预测环境、实测元素数、最高概率、次高环境、前两类概率差、归一化熵。

## 化学相似度检索（v1.1.0 新增）

每次分析同时对照 16,455 条带文献与地区标注的 D36 训练样品库（expansion13），按**实测元素**计算掩码 log10-RMS 距离（相似指数 = 100×10^(−距离)，越高越相似），给出三个视图：

- **相似样品 Top20**：整体最近的单个训练样品（编号、样品名、类别、地区、文献、来源链接）；
- **地区相似排名 Top20**：按地区聚合（组内最近 3 个样品均值再对全部查询样品平均）；
- **文献相似排名 Top20**：按文献出处聚合，同上。

检索基于化学空间距离，仅表示成分相似性，不构成地理或文献来源的概率判定。三个表均随 Excel 一并导出。

**指标解释**：对每条查询样品，只在其实测元素上计算与检索库样品的 log10 浓度均方根（RMS）距离，再对全部查询样品取平均，得到 log10_RMS 距离；相似指数 = 100×10^(−距离)，100 表示完全相同。地区/文献排名采用两级聚合：每组取组内最近的 3 个样品求距离均值，再对全部查询样品平均后排序。

## 界面操作

- 所有表格支持底部**横向滚动条**与右侧纵向滚动条；
- 选中行后按 **Ctrl+C** 复制（带表头，TSV 格式，可直接粘贴到 Excel）；
- Ctrl+点击可多选。

## 界面标识

- 窗口图标使用 `assets/app_icon.ico`；如需更换，替换该文件后重新打包即可。

## 可移植性

- **GPU 版**（约 8.3 GB，CUDA torch）：拷贝整个文件夹到 Windows 10/11 x64 机器即可，无需安装 Python；需要 NVIDIA 显卡 + CUDA 驱动；
- **CPU 版**（约 1 GB，`tools/build_cpu.py` 构建）：任何 Windows x64 机器可运行，无需显卡；
- 另有**网页 demo**（`web/`，FastAPI + 单页前端）：本机运行 `run.bat` 或 `npm run dev` 后浏览器访问 `http://localhost:7100/`，功能与桌面版一致；推理核心与桌面版完全相同。

## 说明

- 概率解释应结合样品地质背景；本工具是研究软件，输出不构成确定性的成因判定。
- 参考上下文、检查点与上游推理代码均已冻结并本地打包（校验和见 assets 元数据）。
- 与 D36 工具（GeoSinglePrior-D36-v2，LightGBM）互为对照；两者类别定义一致。

## 从源码构建 CPU 版

使用独立的 Windows x64 Python 3.12 环境，安装 `requirements-cpu.txt`，准备上述冻结资源后运行：

```powershell
python -m pip install -r apps/limix2m_predictor/requirements-cpu.txt
python -m unittest discover -s apps/limix2m_predictor/tests -v
python apps/limix2m_predictor/tools/build_cpu.py
```

构建脚本核对 CPU 运行库，并运行 EXE 冻结概率回放与界面分析/导出检查。构建检查不构成独立地质验证准确率。
