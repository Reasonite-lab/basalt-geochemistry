# 第三方组件与地球化学资料来源

软件项目代码采用仓库中的 Apache License 2.0。随程序分发的 Python、Tcl/Tk、NumPy、SciPy、pandas、LightGBM、openpyxl 等组件保留其各自许可证，见 `third_party_licenses/`。

模型为项目已冻结的 GeoSinglePrior-D36-v1 三成员 LightGBM 集成，模型训练样品 16,076 件。检索库为另一冻结批次，共 16,346 件、234 个地区、1,689 组原始文献。训练集与检索库并非同一批次。

检索库来源标记包括 GEOROC、EarthChem、Mendeley Data 文献数据集和 Zenodo 汇编。具体样品的原始文献及数据来源链接保存在检索索引中，并随相似度结果导出；数据库主页为：

- GEOROC：https://georoc.eu/
- EarthChem：https://www.earthchem.org/
- Mendeley Data：https://data.mendeley.com/
- Zenodo：https://zenodo.org/

项目代码许可证不替代来源数据库、原始文献或数据集的使用条件。引用或再次分发数据时应依据导出结果中的来源链接，保留原始引用并核对各来源条款。

软件不收集或上传用户输入，全部分析在本机执行。九类概率来自冻结研究模型；相似度仅描述化学距离，不是原产地区或文献来源的概率。
