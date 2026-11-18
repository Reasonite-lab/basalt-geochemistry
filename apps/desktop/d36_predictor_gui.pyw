"""Chinese desktop GUI for D36 tectonic-setting probabilities and retrieval."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from d36_predictor_core import (
    APP_DIR,
    BUNDLE_DIR,
    CLASS_CN,
    CLASSES,
    D36Engine,
    E36,
    AnalysisResult,
    canonicalize_input,
    default_output_path,
    export_result,
    read_input_table,
    write_blank_template,
)


BG = "#F4F7F9"
PANEL = "#FFFFFF"
NAVY = "#17324D"
BLUE = "#2F6B8A"
TEAL = "#3A8D8C"
ORANGE = "#F39C34"
MUTED = "#627383"
LINE = "#D8E1E8"
PALE_BLUE = "#EAF2F7"
PALE_ORANGE = "#FFF3E4"


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
            header,
            text="手工录入 D36 元素",
            font=("Microsoft YaHei UI", 19, "bold"),
            fg="white",
            bg=NAVY,
        ).pack(anchor="w", padx=24, pady=(12, 0))
        tk.Label(
            header,
            text="所有36个元素必须为正数；经纬度可以留空。",
            font=("Microsoft YaHei UI", 10),
            fg="#D8E8F2",
            bg=NAVY,
        ).pack(anchor="w", padx=25, pady=(2, 10))

        meta = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        meta.pack(fill="x", padx=20, pady=14)
        self.meta_entries: dict[str, ttk.Entry] = {}
        meta_items = [("Sample_ID", "样品编号"), ("Latitude", "纬度"), ("Longitude", "经度")]
        for index, (key, label) in enumerate(meta_items):
            tk.Label(meta, text=label, bg=PANEL, fg=NAVY, font=("Microsoft YaHei UI", 10, "bold")).grid(
                row=0, column=index * 2, sticky="e", padx=(12, 6), pady=12
            )
            entry = ttk.Entry(meta, width=22)
            entry.grid(row=0, column=index * 2 + 1, sticky="ew", padx=(0, 14), pady=12)
            self.meta_entries[key] = entry
        meta.grid_columnconfigure(1, weight=1)
        meta.grid_columnconfigure(3, weight=1)
        meta.grid_columnconfigure(5, weight=1)

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
            row = index // 3
            column = (index % 3) * 2
            tk.Label(
                inner,
                text=element,
                bg=PANEL,
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
        row: dict[str, object] = {}
        row["Sample_ID"] = self.meta_entries["Sample_ID"].get().strip() or "manual_0001"
        row["Latitude"] = self.meta_entries["Latitude"].get().strip()
        row["Longitude"] = self.meta_entries["Longitude"].get().strip()
        for element, entry in self.element_entries.items():
            row[element] = entry.get().strip()
        try:
            self.result = canonicalize_input(pd.DataFrame([row]))
        except Exception as error:
            messagebox.showerror("输入检查未通过", str(error), parent=self)
            return
        self.destroy()


class D36App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("D36 未知样品构造环境判别工具")
        self.geometry("1460x900")
        self.minsize(1160, 740)
        self.configure(bg=BG)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self._configure_styles()

        self.engine: D36Engine | None = None
        self.input_table: pd.DataFrame | None = None
        self.result: AnalysisResult | None = None
        self.last_output: Path | None = None
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.geo_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="正在校验并载入冻结模型……")
        self.input_var = tk.StringVar(value="尚未载入样品")
        self.output_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="等待分析")

        self._build_header()
        self._build_toolbar()
        self._build_main()
        self._build_footer()
        self.after(100, self._poll_queue)
        threading.Thread(target=self._load_engine_worker, daemon=True).start()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(18, 10),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("TNotebook.Tab", foreground=[("selected", BLUE)])
        style.configure(
            "Treeview",
            rowheight=30,
            background="white",
            fieldbackground="white",
            foreground="#263746",
            bordercolor=LINE,
        )
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
            background=PALE_BLUE,
            foreground=NAVY,
            relief="flat",
        )
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=NAVY, height=95)
        header.pack(fill="x")
        left = tk.Frame(header, bg=NAVY)
        left.pack(side="left", fill="both", expand=True, padx=28, pady=13)
        tk.Label(
            left,
            text="D36 未知样品构造环境判别工具",
            fg="white",
            bg=NAVY,
            font=("Microsoft YaHei UI", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="九分类概率 · 地区化学相似度 · 文献化学相似度",
            fg="#C9DCE8",
            bg=NAVY,
            font=("Microsoft YaHei UI", 11),
        ).pack(anchor="w", pady=(2, 0))
        badge = tk.Frame(header, bg="#214B68", padx=14, pady=10)
        badge.pack(side="right", padx=28, pady=18)
        tk.Label(
            badge,
            text="冻结研究模型  v1.0",
            fg="white",
            bg="#214B68",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack()
        tk.Label(
            badge,
            text="有效样品不拒识",
            fg="#BFD7E6",
            bg="#214B68",
            font=("Microsoft YaHei UI", 9),
        ).pack()

    def _build_toolbar(self) -> None:
        toolbar = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        toolbar.pack(fill="x", padx=20, pady=(16, 10))
        button_frame = tk.Frame(toolbar, bg=PANEL)
        button_frame.pack(side="left", padx=14, pady=12)
        self.import_button = ttk.Button(button_frame, text="导入 CSV / Excel", command=self._import_file)
        self.import_button.pack(side="left", padx=(0, 8))
        self.manual_button = ttk.Button(button_frame, text="手工录入", command=self._manual_input)
        self.manual_button.pack(side="left", padx=8)
        ttk.Button(button_frame, text="载入示例", command=self._load_example).pack(side="left", padx=8)
        ttk.Button(button_frame, text="导出空白模板", command=self._export_template).pack(side="left", padx=8)
        self.run_button = ttk.Button(
            button_frame,
            text="开始分析",
            command=self._start_analysis,
            style="Accent.TButton",
            state="disabled",
        )
        self.run_button.pack(side="left", padx=(18, 8))

        right = tk.Frame(toolbar, bg=PANEL)
        right.pack(side="right", padx=16, pady=10)
        ttk.Checkbutton(
            right,
            text="使用经纬度空间先验",
            variable=self.geo_var,
        ).pack(anchor="e")
        tk.Label(
            right,
            text="默认关闭；勾选后仅对经纬度完整样品启用",
            bg=PANEL,
            fg=MUTED,
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="e")

    def _build_main(self) -> None:
        main = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6, relief="flat")
        main.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        left = tk.Frame(main, bg=PANEL, highlightbackground=LINE, highlightthickness=1, width=340)
        right = tk.Frame(main, bg=BG)
        main.add(left, minsize=280, width=350)
        main.add(right, minsize=700)
        self._build_input_panel(left)
        self._build_result_tabs(right)

    def _build_input_panel(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="输入样品",
            bg=PANEL,
            fg=NAVY,
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            parent,
            textvariable=self.input_var,
            bg=PANEL,
            fg=MUTED,
            wraplength=300,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))
        columns = ("Sample_ID", "Lat", "Lon")
        self.input_tree = ttk.Treeview(parent, columns=columns, show="headings", height=16)
        self.input_tree.heading("Sample_ID", text="样品编号")
        self.input_tree.heading("Lat", text="纬度")
        self.input_tree.heading("Lon", text="经度")
        self.input_tree.column("Sample_ID", width=150, anchor="w")
        self.input_tree.column("Lat", width=70, anchor="center")
        self.input_tree.column("Lon", width=70, anchor="center")
        self.input_tree.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        info = tk.Frame(parent, bg=PALE_ORANGE, padx=12, pady=10)
        info.pack(fill="x", padx=14, pady=(0, 14))
        tk.Label(
            info,
            text="输入要求",
            bg=PALE_ORANGE,
            fg="#A45A00",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            info,
            text="36个元素必须齐全且为正数。\n经纬度可空缺；每次最多500个样品。",
            bg=PALE_ORANGE,
            fg="#7B5A2A",
            justify="left",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(3, 0))

    def _build_result_tabs(self, parent: tk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        probability_tab = tk.Frame(self.notebook, bg=PANEL)
        region_tab = tk.Frame(self.notebook, bg=PANEL)
        reference_tab = tk.Frame(self.notebook, bg=PANEL)
        audit_tab = tk.Frame(self.notebook, bg=PANEL)
        self.notebook.add(probability_tab, text="① 九分类概率")
        self.notebook.add(region_tab, text="② 地区相似度")
        self.notebook.add(reference_tab, text="③ 文献相似度")
        self.notebook.add(audit_tab, text="说明与审计")
        self._build_probability_tab(probability_tab)
        self.region_tree = self._build_ranking_tab(region_tab, "region")
        self.reference_tree = self._build_ranking_tab(reference_tab, "reference")
        self._build_audit_tab(audit_tab)

    def _build_probability_tab(self, parent: tk.Frame) -> None:
        summary = tk.Frame(parent, bg=PALE_BLUE, padx=16, pady=12)
        summary.pack(fill="x", padx=14, pady=14)
        tk.Label(
            summary,
            textvariable=self.summary_var,
            bg=PALE_BLUE,
            fg=NAVY,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(anchor="w")
        content = tk.PanedWindow(parent, orient="horizontal", bg=PANEL, sashwidth=5)
        content.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        table_frame = tk.Frame(content, bg=PANEL)
        chart_frame = tk.Frame(content, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        content.add(table_frame, minsize=430)
        content.add(chart_frame, minsize=380)

        columns = ("样品", "预测", "中文", "最高概率", "次高", "路径")
        self.probability_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        widths = [145, 60, 120, 85, 60, 210]
        for column, width in zip(columns, widths):
            self.probability_tree.heading(column, text=column)
            self.probability_tree.column(column, width=width, anchor="center" if column != "路径" else "w")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.probability_tree.yview)
        self.probability_tree.configure(yscrollcommand=yscroll.set)
        self.probability_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.probability_tree.bind("<<TreeviewSelect>>", self._on_probability_select)

        self.chart_title = tk.StringVar(value="选择一个样品查看九分类概率")
        tk.Label(
            chart_frame,
            textvariable=self.chart_title,
            bg=PANEL,
            fg=NAVY,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))
        self.probability_canvas = tk.Canvas(chart_frame, bg=PANEL, highlightthickness=0)
        self.probability_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.probability_canvas.bind("<Configure>", lambda _: self._draw_probability_chart())

    def _build_ranking_tab(self, parent: tk.Frame, kind: str) -> ttk.Treeview:
        explanation = (
            "按整组未知样品与训练库各Province的D36化学距离排序。"
            if kind == "region"
            else "按整组未知样品与训练库各Reference的D36化学距离排序。"
        )
        note = tk.Frame(parent, bg=PALE_BLUE, padx=16, pady=12)
        note.pack(fill="x", padx=14, pady=14)
        tk.Label(note, text=explanation, bg=PALE_BLUE, fg=NAVY, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            note,
            text="相似指数用于排序，不是样品来源概率；同时保留log10-RMS距离和近邻证据。",
            bg=PALE_BLUE,
            fg=MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(3, 0))
        columns = (
            ("排名", "地区", "相似指数", "倍数差", "库内N", "证据样品", "类别", "证据文献")
            if kind == "region"
            else ("排名", "文献", "相似指数", "倍数差", "库内N", "证据样品", "类别", "证据地区")
        )
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        widths = [55, 270 if kind == "region" else 430, 90, 85, 70, 120, 70, 360 if kind == "region" else 240]
        for column, width in zip(columns, widths):
            tree.heading(column, text=column)
            tree.column(column, width=width, anchor="center" if column in {"排名", "相似指数", "倍数差", "库内N", "类别"} else "w")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return tree

    def _build_audit_tab(self, parent: tk.Frame) -> None:
        text = (
            "结果解释\n\n"
            "1. 九分类概率来自冻结的三成员 D36_74 LightGBM。每个有效样品都输出九类概率与唯一硬标签，不对测试样品拒识。\n\n"
            "2. 默认使用纯化学路线。只有主动勾选“使用经纬度空间先验”且该样品经纬度完整时，才启用 GeoSinglePrior-D36-v1；坐标缺失自动精确回退 D36_74。\n\n"
            "3. 地区与文献相似度只比较36元素化学向量。先对各元素取log10，再计算RMS距离；每个候选组取最近3个库内样品的平均距离，多未知样品再取平均。相似指数=100×10^(-距离)。\n\n"
            "4. 相似性不能证明样品来自该地区或该文献，也不能替代年代、矿物学、同位素和地质背景证据。\n\n"
            "5. 当前模型属于冻结研究推理包，外部地区确认仍是性能边界。"
        )
        box = tk.Text(
            parent,
            wrap="word",
            bg=PANEL,
            fg="#263746",
            relief="flat",
            padx=26,
            pady=22,
            font=("Microsoft YaHei UI", 11),
        )
        box.pack(fill="both", expand=True, padx=18, pady=18)
        box.insert("1.0", text)
        box.configure(state="disabled")
        self.audit_text = box

    def _build_footer(self) -> None:
        footer = tk.Frame(self, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, textvariable=self.status_var, bg=PANEL, fg=MUTED).pack(
            side="left", padx=20, pady=9
        )
        self.open_button = ttk.Button(footer, text="打开结果文件夹", command=self._open_output_folder, state="disabled")
        self.open_button.pack(side="right", padx=(8, 18), pady=6)
        self.export_button = ttk.Button(footer, text="另存结果", command=self._save_as, state="disabled")
        self.export_button.pack(side="right", padx=8, pady=6)
        tk.Label(footer, textvariable=self.output_var, bg=PANEL, fg=TEAL).pack(
            side="right", padx=10, pady=9
        )

    def _load_engine_worker(self) -> None:
        try:
            engine = D36Engine()
            self.message_queue.put(("engine", engine))
        except Exception:
            self.message_queue.put(("fatal", traceback.format_exc()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "engine":
                    self.engine = payload  # type: ignore[assignment]
                    self.status_var.set("模型与检索库校验通过，可以导入样品。")
                    self.run_button.configure(state="normal" if self.input_table is not None else "disabled")
                elif kind == "result":
                    self._accept_result(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("分析失败；请检查输入。")
                    messagebox.showerror("分析失败", str(payload), parent=self)
                elif kind == "fatal":
                    self.status_var.set("模型加载失败。")
                    messagebox.showerror("启动失败", str(payload), parent=self)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _import_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="选择D36样品表",
            filetypes=[
                ("支持的表格", "*.csv *.txt *.xlsx"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            table = read_input_table(path)
        except Exception as error:
            messagebox.showerror("导入失败", str(error), parent=self)
            return
        self._set_input(table, f"{Path(path).name} · {len(table)}个样品")

    def _manual_input(self) -> None:
        example = None
        try:
            example_frame = pd.read_csv(BUNDLE_DIR / "example_input.csv")
            example_frame = canonicalize_input(example_frame)
            example = None
        except Exception:
            pass
        dialog = ManualInputDialog(self, example=example)
        self.wait_window(dialog)
        if dialog.result is not None:
            self._set_input(dialog.result, "手工录入 · 1个样品")

    def _load_example(self) -> None:
        try:
            frame = pd.read_csv(BUNDLE_DIR / "example_input.csv")
            table = canonicalize_input(frame.head(3))
        except Exception as error:
            messagebox.showerror("示例加载失败", str(error), parent=self)
            return
        self._set_input(table, "冻结模型示例 · 3个样品")

    def _export_template(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="保存D36空白模板",
            defaultextension=".csv",
            initialfile="D36未知样品输入模板.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            write_blank_template(path)
            self.status_var.set(f"模板已保存：{path}")
        except Exception as error:
            messagebox.showerror("模板保存失败", str(error), parent=self)

    def _set_input(self, table: pd.DataFrame, description: str) -> None:
        self.input_table = table
        self.result = None
        self.input_var.set(description)
        for item in self.input_tree.get_children():
            self.input_tree.delete(item)
        for _, row in table.head(200).iterrows():
            latitude = "" if pd.isna(row["Latitude"]) else f"{float(row['Latitude']):.4f}"
            longitude = "" if pd.isna(row["Longitude"]) else f"{float(row['Longitude']):.4f}"
            self.input_tree.insert("", "end", values=(row["Sample_ID"], latitude, longitude))
        self.run_button.configure(state="normal" if self.engine is not None else "disabled")
        self.status_var.set("输入检查通过；点击“开始分析”。")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.import_button.configure(state=state)
        self.manual_button.configure(state=state)
        self.run_button.configure(
            state="disabled" if busy or self.engine is None or self.input_table is None else "normal"
        )
        self.configure(cursor="watch" if busy else "")

    def _start_analysis(self) -> None:
        if self.engine is None:
            messagebox.showinfo("模型正在载入", "请等待模型校验完成。", parent=self)
            return
        if self.input_table is None:
            messagebox.showinfo("没有输入", "请先导入或手工录入样品。", parent=self)
            return
        self._set_busy(True)
        self.status_var.set("正在计算九分类概率和化学相似度……")
        table = self.input_table.copy()
        use_geo = bool(self.geo_var.get())
        threading.Thread(
            target=self._analysis_worker,
            args=(table, use_geo),
            daemon=True,
        ).start()

    def _analysis_worker(self, table: pd.DataFrame, use_geo: bool) -> None:
        try:
            assert self.engine is not None
            result = self.engine.analyze(table, use_spatial_prior=use_geo)
            output_path = default_output_path()
            export_result(result, output_path)
            self.message_queue.put(("result", (result, output_path)))
        except Exception:
            self.message_queue.put(("error", traceback.format_exc()))

    def _accept_result(self, payload: tuple[AnalysisResult, Path]) -> None:
        result, output_path = payload
        self.result = result
        self.last_output = output_path
        self._populate_results()
        self._set_busy(False)
        self.export_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.output_var.set(f"已自动保存：{output_path.name}")
        self.status_var.set("分析完成：九分类概率、地区排名和文献排名均已生成。")

    def _populate_results(self) -> None:
        assert self.result is not None
        for tree in (self.probability_tree, self.region_tree, self.reference_tree):
            for item in tree.get_children():
                tree.delete(item)
        for index, row in self.result.probabilities.iterrows():
            self.probability_tree.insert(
                "",
                "end",
                iid=f"prob_{index}",
                values=(
                    row["样品编号"],
                    row["预测环境"],
                    row["预测环境中文"],
                    f"{float(row['最高概率']):.1%}",
                    row["次高环境"],
                    row["推理路径"],
                ),
            )
        for _, row in self.result.region_ranking.iterrows():
            self.region_tree.insert(
                "",
                "end",
                values=(
                    int(row["排名"]),
                    row["地区"],
                    f"{float(row['相似指数']):.2f}",
                    f"×{float(row['典型倍数差']):.3f}",
                    int(row["检索库样品数"]),
                    row["最近证据样品"],
                    row["最近证据类别"],
                    row["最近证据文献"],
                ),
            )
        for _, row in self.result.reference_ranking.iterrows():
            self.reference_tree.insert(
                "",
                "end",
                values=(
                    int(row["排名"]),
                    row["文献"],
                    f"{float(row['相似指数']):.2f}",
                    f"×{float(row['典型倍数差']):.3f}",
                    int(row["检索库样品数"]),
                    row["最近证据样品"],
                    row["最近证据类别"],
                    row["最近证据地区"],
                ),
            )
        first = self.result.probabilities.iloc[0]
        self.summary_var.set(
            f"已分析 {len(self.result.probabilities)} 个样品；首个样品预测为 "
            f"{first['预测环境']}（{first['预测环境中文']}），最高概率 {float(first['最高概率']):.1%}。"
        )
        children = self.probability_tree.get_children()
        if children:
            self.probability_tree.selection_set(children[0])
            self.probability_tree.focus(children[0])
            self._draw_probability_chart(0)

    def _on_probability_select(self, _: object = None) -> None:
        selection = self.probability_tree.selection()
        if not selection:
            return
        try:
            index = int(selection[0].split("_")[-1])
        except (ValueError, IndexError):
            return
        self._draw_probability_chart(index)

    def _draw_probability_chart(self, index: int | None = None) -> None:
        canvas = self.probability_canvas
        canvas.delete("all")
        if self.result is None:
            return
        if index is None:
            selection = self.probability_tree.selection()
            if selection:
                try:
                    index = int(selection[0].split("_")[-1])
                except (ValueError, IndexError):
                    index = 0
            else:
                index = 0
        row = self.result.probabilities.iloc[int(index)]
        values = np.asarray([float(row[f"P_{name}"]) for name in CLASSES])
        self.chart_title.set(f"{row['样品编号']} · {row['预测环境']} {row['预测环境中文']}")
        width = max(canvas.winfo_width(), 440)
        height = max(canvas.winfo_height(), 420)
        left = 88
        right = 58
        top = 18
        usable_width = width - left - right
        row_height = (height - top - 20) / len(CLASSES)
        max_value = max(values.max(), 0.35)
        colors = [ORANGE if name == row["预测环境"] else BLUE for name in CLASSES]
        for rank, (name, value, color) in enumerate(zip(CLASSES, values, colors)):
            y = top + rank * row_height
            canvas.create_text(
                left - 8,
                y + row_height * 0.42,
                text=f"{name}  {CLASS_CN[name]}",
                anchor="e",
                fill=NAVY,
                font=("Microsoft YaHei UI", 9, "bold" if name == row["预测环境"] else "normal"),
            )
            canvas.create_rectangle(
                left,
                y + row_height * 0.18,
                left + usable_width,
                y + row_height * 0.67,
                fill="#EEF2F5",
                outline="",
            )
            bar_width = usable_width * value / max_value
            canvas.create_rectangle(
                left,
                y + row_height * 0.18,
                left + bar_width,
                y + row_height * 0.67,
                fill=color,
                outline="",
            )
            canvas.create_text(
                min(left + bar_width + 8, width - 45),
                y + row_height * 0.42,
                text=f"{value:.1%}",
                anchor="w",
                fill=color if value > 0.01 else MUTED,
                font=("Segoe UI", 9, "bold"),
            )

    def _save_as(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="另存完整分析结果",
            defaultextension=".xlsx",
            initialfile=self.last_output.name if self.last_output else "D36未知样品分析.xlsx",
            filetypes=[("Excel工作簿", "*.xlsx")],
        )
        if not path:
            return
        try:
            export_result(self.result, path)
            self.last_output = Path(path)
            self.output_var.set(f"已另存：{Path(path).name}")
        except Exception as error:
            messagebox.showerror("保存失败", str(error), parent=self)

    def _open_output_folder(self) -> None:
        folder = self.last_output.parent if self.last_output is not None else APP_DIR / "outputs"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            subprocess.Popen(["explorer", str(folder)])


def main() -> None:
    if "--self-test" in sys.argv:
        input_path: Path | None = None
        output_dir = APP_DIR / "portable_self_test"
        if "--input" in sys.argv:
            input_path = Path(sys.argv[sys.argv.index("--input") + 1]).resolve()
        if "--output-dir" in sys.argv:
            output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = read_input_table(input_path) if input_path else pd.read_csv(BUNDLE_DIR / "example_input.csv")
        engine = D36Engine()
        result = engine.analyze(frame, use_spatial_prior=False)
        output_path = output_dir / "portable_self_test_result.xlsx"
        export_result(result, output_path)
        probability = result.probabilities[[f"P_{name}" for name in CLASSES]].to_numpy(float)
        report = {
            "status": "PASS",
            "frozen_executable": bool(getattr(sys, "frozen", False)),
            "executable": str(Path(sys.executable).resolve()),
            "resource_dir": str(Path(getattr(sys, "_MEIPASS", APP_DIR)).resolve()),
            "input": str(input_path) if input_path else "bundled example_input.csv",
            "input_rows": int(len(frame)),
            "output": str(output_path),
            "audit_output": str(output_path.with_suffix(".audit.json")),
            "probability_shape": [int(value) for value in probability.shape],
            "probabilities_sum_to_one": bool(np.allclose(probability.sum(axis=1), 1.0, atol=1e-12)),
            "all_rows_labeled": bool(result.probabilities["预测环境"].notna().all()),
            "region_ranking_rows": int(len(result.region_ranking)),
            "reference_ranking_rows": int(len(result.reference_ranking)),
        }
        report_path = output_dir / "portable_self_test_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not (
            report["frozen_executable"]
            and report["probabilities_sum_to_one"]
            and report["all_rows_labeled"]
            and output_path.exists()
            and output_path.with_suffix(".audit.json").exists()
        ):
            raise SystemExit(1)
        return
    app = D36App()
    app.mainloop()


if __name__ == "__main__":
    main()

