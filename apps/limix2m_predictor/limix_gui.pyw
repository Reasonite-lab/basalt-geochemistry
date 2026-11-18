"""LimiX-2M 构造环境判别工具 — desktop GUI (Windows, Tk)."""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from version import APP_NAME, APP_VERSION, MODEL_NAME
from limix_core import (
    APP_DIR,
    ASSET_DIR,
    CLASSES,
    E36,
    LimiX2MEngine,
    canonicalize_input,
    default_output_path,
    export_result,
    read_input_table,
    write_blank_template,
)

BG = "#F4F7FA"
PANEL = "#FFFFFF"
NAVY = "#1B3A53"
ACCENT = "#3A8D8C"
MUTED = "#627383"
LINE = "#D8E1E8"
PALE_BLUE = "#EAF2F7"


class ManualInputDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, example: pd.Series | None = None) -> None:
        super().__init__(parent)
        self.title("手工录入一个未知样品")
        self.geometry("920x760")
        self.minsize(820, 640)
        self.transient(parent)
        self.grab_set()
        self.result: pd.DataFrame | None = None
        self.configure(bg=BG)

        header = tk.Frame(self, bg=NAVY, height=72)
        header.pack(fill="x")
        tk.Label(
            header, text="手工录入 D36 元素",
            font=("Microsoft YaHei UI", 19, "bold"), fg="white", bg=NAVY,
        ).pack(anchor="w", padx=24, pady=(12, 0))
        tk.Label(
            header,
            text="元素留空表示缺测；最低要求全部10项主量加8项微量（共≥18项）。",
            font=("Microsoft YaHei UI", 10), fg="#D8E8F2", bg=NAVY,
        ).pack(anchor="w", padx=25, pady=(2, 10))

        meta = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        meta.pack(fill="x", padx=20, pady=14)
        self.meta_entries: dict[str, ttk.Entry] = {}
        for index, (key, label) in enumerate(
            [("Sample_ID", "样品编号"), ("Latitude", "纬度"), ("Longitude", "经度")]
        ):
            tk.Label(meta, text=label, bg=PANEL, fg=NAVY,
                     font=("Microsoft YaHei UI", 10, "bold")).grid(
                row=0, column=index * 2, sticky="e", padx=(12, 6), pady=12)
            entry = ttk.Entry(meta, width=22)
            entry.grid(row=0, column=index * 2 + 1, sticky="ew", padx=(0, 14), pady=12)
            self.meta_entries[key] = entry
        for column in (1, 3, 5):
            meta.grid_columnconfigure(column, weight=1)

        holder = tk.Frame(self, bg=BG)
        holder.pack(fill="both", expand=True, padx=20)
        canvas = tk.Canvas(holder, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.element_entries: dict[str, ttk.Entry] = {}
        for index, element in enumerate(E36):
            row, column = index // 3, (index % 3) * 2
            tk.Label(
                inner, text=element, bg=PANEL,
                fg=NAVY if index < 10 else MUTED,
                font=("Segoe UI", 9, "bold" if index < 10 else "normal"),
            ).grid(row=row, column=column, sticky="e", padx=(12, 5), pady=7)
            entry = ttk.Entry(inner, width=16)
            entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 15), pady=7)
            self.element_entries[element] = entry
        for column in (1, 3, 5):
            inner.grid_columnconfigure(column, weight=1)

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=20, pady=16)
        ttk.Button(footer, text="取消", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(footer, text="加入分析", command=self._accept, style="Accent.TButton").pack(side="right")
        if example is not None:
            self._fill(example)

    def _fill(self, example: pd.Series) -> None:
        for key, entry in self.meta_entries.items():
            if key in example and pd.notna(example[key]):
                entry.insert(0, str(example[key]))
        for element, entry in self.element_entries.items():
            if element in example and pd.notna(example[element]):
                entry.insert(0, str(example[element]))

    def _accept(self) -> None:
        row: dict[str, object] = {"Sample_ID": self.meta_entries["Sample_ID"].get().strip() or "manual_0001",
                                  "Latitude": self.meta_entries["Latitude"].get().strip(),
                                  "Longitude": self.meta_entries["Longitude"].get().strip()}
        for element, entry in self.element_entries.items():
            row[element] = entry.get().strip()
        try:
            self.result = canonicalize_input(pd.DataFrame([row]))
        except Exception as error:
            messagebox.showerror("输入检查未通过", str(error), parent=self)
            return
        self.destroy()


class LimiXApp(tk.Tk):
    def __init__(self, engine: LimiX2MEngine | None = None) -> None:
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1320x860")
        self.minsize(1080, 700)
        self.configure(bg=BG)
        self.engine = engine
        self.pending_frames: list[pd.DataFrame] = []
        self.last_result = None
        icon_path = ASSET_DIR / "app_icon.ico"
        if icon_path.exists():
            try:
                self.iconbitmap(str(icon_path))
            except tk.TclError:
                pass

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", foreground="white", background=ACCENT)
        style.map("Accent.TButton", background=[("active", "#2F7473")])

        top = tk.Frame(self, bg=NAVY, height=86)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(
            top, text=APP_NAME, font=("Microsoft YaHei UI", 20, "bold"),
            fg="white", bg=NAVY,
        ).pack(side="left", padx=24, pady=(10, 0))
        badge = tk.Frame(top, bg="#214B68")
        badge.pack(side="right", padx=(0, 20), pady=18)
        tk.Label(
            badge, text=MODEL_NAME, fg="white", bg="#214B68",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(padx=14, pady=6)

        toolbar = tk.Frame(self, bg=BG)
        toolbar.pack(fill="x", padx=20, pady=(14, 8))
        for text, command in [
            ("导入 CSV/XLSX", self.import_file),
            ("手工录入", self.manual_entry),
            ("载入示例", self.load_example),
            ("导出 Excel", self.export_excel),
            ("空白模板", self.save_template),
        ]:
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=(0, 8))
        self.status_var = tk.StringVar(
            value="就绪。选中表格行后按 Ctrl+C 可复制（可直接粘贴到 Excel）；表格支持底部横向滚动。")
        tk.Label(
            toolbar, textvariable=self.status_var, bg=BG, fg=MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right")

        table_frame = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        table_frame.pack(fill="both", expand=True, padx=20, pady=(4, 20))
        self.notebook = ttk.Notebook(table_frame)
        self.notebook.pack(fill="both", expand=True)

        prob_columns = ["样品编号", "预测环境", "预测环境中文", "实测元素数", "最高概率",
                        "次高环境", "前两类概率差", "归一化熵"] + [f"P_{c}" for c in CLASSES]
        self.tree = self._make_table_tab(
            "九分类概率", prob_columns,
            widths={"样品编号": 200, "预测环境中文": 110},
            default_width=90,
        )

        sample_columns = ["排名", "样品编号", "样品名", "类别", "类别中文", "地区", "文献",
                          "来源库", "相似指数", "log10_RMS距离", "来源链接"]
        self.sample_tree = self._make_table_tab(
            "相似样品 Top20", sample_columns,
            widths={"样品编号": 170, "样品名": 220, "文献": 260, "来源链接": 320},
            default_width=90,
            note=("相似指数与 log10_RMS 距离：对每条查询样品，只在其实测元素上计算与检索库样品的 "
                  "log10 浓度均方根距离，再对全部查询样品取平均。相似指数 = 100×10^(−距离)，"
                  "100 表示完全相同，越小差异越大。来源链接指向原始数据出处（EarthChem/GEOROC 等）。"),
        )

        ranking_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(ranking_tab, text="地区/文献相似排名")
        group_columns = ["排名", "名称", "相似指数", "log10_RMS距离",
                         "检索库样品数", "最近证据样品", "最近证据类别", "来源链接"]
        self.region_tree = self._make_table(
            ranking_tab, "地区相似排名（Top20）", group_columns,
            widths={"名称": 240, "最近证据样品": 200, "来源链接": 300},
            default_width=100,
        )
        self.reference_tree = self._make_table(
            ranking_tab, "文献相似排名（Top20）", group_columns,
            widths={"名称": 420, "最近证据样品": 200, "来源链接": 300},
            default_width=100,
        )
        tk.Label(
            ranking_tab,
            text=("排名算法：对每个地区/文献组，取组内与查询距离最近的 3 个样品求距离均值，"
                  "再对全部查询样品取平均后排序。检索基于化学空间距离，仅表示成分相似性，"
                  "不构成地理或文献来源的概率判定。"),
            bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 9), wraplength=1180, justify="left",
        ).pack(fill="x", padx=14, pady=(0, 10))

    def _make_table(self, parent: tk.Misc, title: str, columns: list[str],
                    widths: dict[str, int] | None = None,
                    default_width: int = 90, note: str = "") -> ttk.Treeview:
        holder = tk.Frame(parent, bg=PANEL)
        if title:
            tk.Label(holder, text=title, bg=PANEL, fg=NAVY,
                     font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 4))
        frame = tk.Frame(holder, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        for name in columns:
            tree.heading(name, text=name)
            tree.column(name, width=(widths or {}).get(name, default_width),
                        anchor="center", stretch=False)
        tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        xscroll = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=xscroll.set)
        xscroll.pack(fill="x", padx=6, pady=(0, 6))
        if note:
            tk.Label(holder, text=note, bg=PANEL, fg=MUTED,
                     font=("Microsoft YaHei UI", 9), wraplength=1160,
                     justify="left").pack(fill="x", padx=10, pady=(0, 8))
        tree.bind("<Control-c>", lambda _event, t=tree: self._copy_tree(t))
        tree.bind("<Control-C>", lambda _event, t=tree: self._copy_tree(t))
        holder.pack(fill="both", expand=True, padx=10, pady=10)
        return tree

    def _make_table_tab(self, tab_title: str, columns: list[str],
                        widths: dict[str, int] | None = None,
                        default_width: int = 90, note: str = "") -> ttk.Treeview:
        tab = tk.Frame(self.notebook, bg=PANEL)
        self.notebook.add(tab, text=tab_title)
        return self._make_table(tab, "", columns, widths, default_width, note)

    def _copy_tree(self, tree: ttk.Treeview) -> None:
        selection = tree.selection()
        if not selection:
            self.status_var.set("请先选中要复制的行（Ctrl+点击可多选）。")
            return
        lines = ["\t".join(str(name) for name in tree["columns"])]
        for item in selection:
            lines.append("\t".join(str(value) for value in tree.item(item, "values")))
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.status_var.set(f"已复制 {len(selection)} 行到剪贴板（可直接粘贴到 Excel）。")

    @staticmethod
    def _fill_tree(tree: ttk.Treeview, frame: pd.DataFrame) -> None:
        tree.delete(*tree.get_children())
        for _, row in frame.iterrows():
            values = []
            for name in tree["columns"]:
                value = row[name]
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            tree.insert("", "end", values=values)

    def _ensure_engine(self) -> LimiX2MEngine:
        if self.engine is None:
            self.status_var.set("正在加载 LimiX-2M 模型…")
            self.update_idletasks()
            self.engine = LimiX2MEngine()
            self.status_var.set(f"模型已加载（{self.engine.device.upper()}）。")
        return self.engine

    def _run_analysis(self, frame: pd.DataFrame) -> None:
        try:
            engine = self._ensure_engine()
            self.status_var.set(f"正在分析 {len(frame)} 个样品…")
            self.update_idletasks()
            result = engine.analyze(frame)
        except Exception:
            messagebox.showerror("分析失败", traceback.format_exc(), parent=self)
            self.status_var.set("分析失败。")
            return
        self.last_result = result
        self._fill_tree(self.tree, result.probabilities)
        self._fill_tree(self.sample_tree, result.nearest_samples)
        self._fill_tree(self.region_tree, result.region_ranking)
        self._fill_tree(self.reference_tree, result.reference_ranking)
        audit = result.audit
        self.status_var.set(
            f"完成：{audit['input_rows']} 个样品，{audit['inference_seconds']:.1f}s，"
            f"设备 {audit['device']}，上下文 {audit['context_budget']} 条，"
            f"检索库 {audit['retrieval_library_rows']} 条。"
        )

    def import_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            frame = read_input_table(path)
        except Exception as error:
            messagebox.showerror("导入失败", str(error), parent=self)
            return
        self._run_analysis(frame)

    def manual_entry(self) -> None:
        dialog = ManualInputDialog(self)
        self.wait_window(dialog)
        if dialog.result is not None:
            self._run_analysis(dialog.result)

    def load_example(self) -> None:
        path = ASSET_DIR / "example_input.csv"
        if not path.exists():
            messagebox.showerror("示例缺失", f"未找到示例文件：{path}", parent=self)
            return
        try:
            frame = read_input_table(path)
        except Exception as error:
            messagebox.showerror("示例载入失败", str(error), parent=self)
            return
        self._run_analysis(frame)

    def export_excel(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("无可导出结果", "请先完成一次分析。", parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="导出 Excel", defaultextension=".xlsx",
            initialfile=default_output_path().name,
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            export_result(self.last_result, path)
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc(), parent=self)
            return
        self.status_var.set(f"已导出：{path}")

    def save_template(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存空白模板", defaultextension=".csv",
            initialfile="LimiX2M未知样品输入模板.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        write_blank_template(path)
        self.status_var.set(f"模板已保存：{path}")


def run_self_test(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ASSET_DIR / "example_input.csv")
    engine = LimiX2MEngine()
    result = engine.analyze(frame)
    output_path = output_dir / "portable_self_test_result.xlsx"
    export_result(result, output_path)
    probability = result.probabilities[[f"P_{c}" for c in CLASSES]].to_numpy(float)
    replay_error = None
    expected = pd.read_csv(ASSET_DIR / "example_predictions.csv")
    replay_error = float(np.max(np.abs(
        probability - expected[[f"prob_{c}" for c in CLASSES]].to_numpy(float)
    )))
    if replay_error > 1e-3:
        raise RuntimeError(f"Frozen probability replay mismatch: {replay_error}")
    labels_match = bool(
        (probability.argmax(axis=1)
         == expected[[f"prob_{c}" for c in CLASSES]].to_numpy(float).argmax(axis=1)).all()
    )
    report = {
        "status": "PASS",
        "version": APP_VERSION,
        "model": MODEL_NAME,
        "device": engine.device,
        "frozen_probability_replay_max_abs_error": replay_error,
        "frozen_labels_match": labels_match,
        "frozen_executable": bool(getattr(sys, "frozen", False)),
        "input_rows": int(len(frame)),
        "output": str(output_path),
        "probabilities_sum_to_one": bool(np.allclose(probability.sum(axis=1), 1.0, atol=1e-9)),
    }
    (output_dir / "portable_self_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not (labels_match and report["probabilities_sum_to_one"] and output_path.exists()):
        raise SystemExit(1)


def run_gui_smoke_test(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    app = LimiXApp(engine=None)
    app.withdraw()

    def wait_until(predicate, label: str) -> None:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            app.update()
            if predicate():
                return
            time.sleep(0.05)
        raise TimeoutError(f"GUI smoke timeout: {label}")

    try:
        app.load_example()
        wait_until(lambda: app.last_result is not None, "example analysis")
        export_path = output_dir / "gui_smoke_export.xlsx"
        app.last_result and export_result(app.last_result, export_path)
        report = {
            "status": "PASS",
            "frozen_executable": bool(getattr(sys, "frozen", False)),
            "probability_rows": len(app.tree.get_children()),
            "nearest_sample_rows": len(app.sample_tree.get_children()),
            "region_ranking_rows": len(app.region_tree.get_children()),
            "reference_ranking_rows": len(app.reference_tree.get_children()),
            "excel_export": export_path.exists(),
        }
        (output_dir / "gui_smoke_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not (report["probability_rows"] and report["nearest_sample_rows"]
                and report["region_ranking_rows"] and report["reference_ranking_rows"]
                and report["excel_export"]):
            raise SystemExit(1)
    finally:
        app.destroy()


def main() -> None:
    if "--self-test" in sys.argv:
        output_dir = APP_DIR / "portable_self_test"
        if "--output-dir" in sys.argv:
            output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
        run_self_test(output_dir)
        return
    if "--gui-smoke-test" in sys.argv:
        output_dir = APP_DIR / "portable_gui_test"
        if "--output-dir" in sys.argv:
            output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
        run_gui_smoke_test(output_dir)
        return
    app = LimiXApp()
    app.mainloop()


if __name__ == "__main__":
    main()
