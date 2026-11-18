# 第三方组件与数据来源

本项目代码采用 Apache License 2.0，见 `LICENSE`。冻结的 LimiX-2M 上游代码与检查点保留其 Apache-2.0 许可，见 `LimiX_LICENSE.txt`。这指本版使用的 LimiX-2M，不泛指其他后续同名模型。

程序同时包含 Python、Tcl/Tk、PyTorch、NumPy、pandas、SciPy、scikit-learn、Numba、llvmlite、einops、openpyxl 等组件，各自的许可证及依赖版本见 `third_party_licenses/`。冻结资源中的 kditransform 等依赖保留各自的原始许可。

参考上下文与检索库来自项目 expansion13 冻结汇编。地球化学资料的原始数据库、文献与来源链接随检索结果导出，代码许可证不替代各来源资料的使用条件。使用数据时须保留原始引用并核对其来源条款。

分析在本机执行，程序不会上传用户输入。检索相似度是化学距离的展示，不是原产地区或文献来源概率。
