# 玄武岩地球化学构造环境判别

保存玄武岩构造环境判别研究及桌面预测工具的代码，作为后续实时预测软件与网站的开发基础。

当前支持的九类构造标签为 BAB、CA、CFB、CR、IA、IOA、MOR、OI 和 OP。研究包含全岩元素、真实缺测、人工遮挡及 Sr–Nd–Pb 同位素输入的配对比较。

## 代码结构

| 目录 | 内容 |
|---|---|
| `src/` | 本轮 LimiX-2M、LightGBM、XGBoost、CatBoost、随机森林的实验、审计与出图，以及所需项目模块 |
| `scripts/` | AFM 和稀土元素图的绘图依赖 |
| `apps/desktop/` | 已有桌面预测器的界面、预测核心、资源准备程序及输入模板 |
| `apps/limix2m_predictor/` | LimiX-2M 桌面工具、CPU 打包入口及共享推理核心的本地网页 demo |
| `presentation/` | 配套研究汇报的生成程序 |
| `docs/` | 预测软件与网站的开发安排、源文件校验清单 |

## 安装与运行

建议 Python 3.12，在仓库根目录执行。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

研究运行需要项目的冻结数据、样品清单和质控文件，按照各入口中的路径放在 `data/`、`reports/` 等本地目录。LimiX-2M 还需要官方推理代码、配置、对应依赖与检查点，通过命令行参数指定位置。原始实验代码及其输入文件名保留了研究日期，具体代码映射见 `docs/source-manifest.json`。

```powershell
.venv/Scripts/python src/run_limix2m_geology_paper.py --help
.venv/Scripts/python src/run_limix2m_geology_isotopes.py --help
```

桌面程序使用已保存的 LightGBM 研究模型与检索库。将所需资源放入 `apps/desktop/assets/` 后运行。

```powershell
.venv/Scripts/python apps/desktop/d36_predictor_gui.pyw
```

已有两套桌面工具：D36 使用 GeoSinglePrior-D36-v2（LightGBM），LimiX-2M 使用冻结参考上下文推理。两者都支持缺测输入，最低要求全部 10 项主量加至少 8 项微量元素，最多一次处理 500 件样品。LimiX-2M 的本地网页 demo 与桌面版共用推理核心，运行方法见 `apps/limix2m_predictor/README.md`。

Windows 发行包见 [GitHub Releases](https://github.com/Reasonite-lab/basalt-geochemistry/releases)。LimiX-2M 采用 CPU 便携包；GPU 版仅保留本地构建。

## 当前实验入口

- `src/run_limix2m_geology_paper.py` 比较元素面板与带标签参考样品数量。
- `src/run_limix2m_geology_isotopes.py` 比较 Sr–Nd–Pb 实测值与可用性标记。
- `src/run_limix2m_panel_control.py` 在保留共同 22 项测量时比较遮挡组合，记录树模型拟合过程。
- `src/run_limix2m_geology_baselines.py` 使用相同的参考样品比较五种模型。
- `src/run_limix2m_geology_expansion.py` 汇总扩展真实缺测队列的对照。
- `src/audit_limix2m_geology_paper.py` 核对样品索引和预测概率。

当前仓库保存代码和空白输入模板。原始资料、模型权重、检索库、论文成品及运行环境由本地项目管理。数据分享与模型分发需沿用各来源的使用条件。

## 后续实时预测

桌面端与网站将共用字段、单位、缺测处理和模型版本规范。开发顺序见 `docs/roadmap.md`。

实验结果对应库内随机验证，软件输出的类别概率应结合样品背景和区域地质使用。
