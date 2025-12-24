"""
科研图表生成对话框
提供地质、机器学习、结果分析等专业图表的生成和导出
"""

import os
from typing import Dict, Optional, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QGroupBox, QCheckBox,
    QFileDialog, QMessageBox, QProgressBar, QTextEdit,
    QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np


class ChartGenerationThread(QThread):
    """图表生成线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, chart_type: str, data: Dict, options: Dict):
        super().__init__()
        self.chart_type = chart_type
        self.data = data
        self.options = options

    def run(self):
        try:
            from src.visualization import (
                GeologyPlots, MLPlots, ResultPlots,
                SCIFigureStyle, FigureExporter
            )

            self.progress.emit(f"正在生成 {self.chart_type} 图表...")

            style = SCIFigureStyle()

            # 根据类型选择绘图器
            if 'borehole_layout' in self.chart_type:
                plotter = GeologyPlots(style)
                df = self.data.get('raw_df')
                fig = plotter.plot_borehole_layout(
                    df,
                    return_plotly=self.options.get('use_plotly', False),
                    show_labels=self.options.get('show_labels', True),
                    show_convex_hull=self.options.get('show_convex_hull', True)
                )

            elif 'stratigraphic_correlation' in self.chart_type:
                plotter = GeologyPlots(style)
                df = self.data.get('raw_df')
                fig = plotter.plot_stratigraphic_correlation(
                    df,
                    max_boreholes=self.options.get('max_boreholes', 8),
                    return_plotly=self.options.get('use_plotly', False)
                )

            elif 'thickness_contour' in self.chart_type:
                plotter = GeologyPlots(style)
                df = self.data.get('raw_df')
                lithology = self.options.get('lithology', None)
                fig = plotter.plot_thickness_contour(
                    df,
                    lithology=lithology,
                    resolution=self.options.get('resolution', 50),
                    return_plotly=self.options.get('use_plotly', False)
                )

            elif 'stratigraphic_column' in self.chart_type:
                plotter = GeologyPlots(style)
                df = self.data.get('raw_df')
                borehole_id = self.options.get('borehole_id')
                if not borehole_id:
                    borehole_id = df['borehole_id'].iloc[0]
                fig = plotter.plot_stratigraphic_column(
                    df,
                    borehole_id=borehole_id,
                    return_plotly=self.options.get('use_plotly', False)
                )

            elif 'fence_diagram' in self.chart_type:
                plotter = GeologyPlots(style)
                df = self.data.get('raw_df')
                fig = plotter.plot_fence_diagram(
                    df,
                    geo_model=None,
                    return_plotly=True
                )

            else:
                raise ValueError(f"未知的图表类型: {self.chart_type}")

            self.progress.emit("图表生成完成")
            self.finished.emit({'figure': fig, 'type': self.chart_type})

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class GeologyChartDialog(QDialog):
    """地质专业图表对话框"""

    def __init__(self, parent, data_result: Dict):
        super().__init__(parent)
        self.data_result = data_result
        self.current_figure = None

        self.setWindowTitle("地质专业图表生成器")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("📊 地质专业图表生成器")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # 图表列表
        list_group = QGroupBox("选择图表类型")
        list_layout = QVBoxLayout()

        self.chart_list = QListWidget()
        self.chart_list.addItem("🗺️ 钻孔布置平面图 (Borehole Layout)")
        self.chart_list.addItem("📊 地层对比图 (Stratigraphic Correlation)")
        self.chart_list.addItem("📈 厚度等值线图 (Thickness Contour)")
        self.chart_list.addItem("📏 地层柱状图 (Stratigraphic Column)")
        self.chart_list.addItem("🎲 三维栅栏图 (3D Fence Diagram)")
        self.chart_list.setCurrentRow(0)
        list_layout.addWidget(self.chart_list)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # 选项
        options_group = QGroupBox("图表选项")
        options_layout = QVBoxLayout()

        self.use_plotly_cb = QCheckBox("使用Plotly交互式图表 (推荐)")
        self.use_plotly_cb.setChecked(True)
        options_layout.addWidget(self.use_plotly_cb)

        self.show_labels_cb = QCheckBox("显示标签")
        self.show_labels_cb.setChecked(True)
        options_layout.addWidget(self.show_labels_cb)

        # 分辨率选项
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("插值分辨率:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(20, 200)
        self.resolution_spin.setValue(50)
        res_layout.addWidget(self.resolution_spin)
        res_layout.addStretch()
        options_layout.addLayout(res_layout)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)

        # 按钮
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("🎨 生成图表")
        self.generate_btn.clicked.connect(self.generate_chart)
        btn_layout.addWidget(self.generate_btn)

        self.export_btn = QPushButton("💾 导出图表")
        self.export_btn.clicked.connect(self.export_chart)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def log(self, message: str):
        """添加日志"""
        self.log_text.append(message)

    def generate_chart(self):
        """生成图表"""
        selected = self.chart_list.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "警告", "请选择一个图表类型")
            return

        chart_types = [
            'borehole_layout',
            'stratigraphic_correlation',
            'thickness_contour',
            'stratigraphic_column',
            'fence_diagram'
        ]

        chart_type = chart_types[selected]

        # 准备选项
        options = {
            'use_plotly': self.use_plotly_cb.isChecked(),
            'show_labels': self.show_labels_cb.isChecked(),
            'resolution': self.resolution_spin.value(),
        }

        # 特殊处理厚度等值线图 - 需要选择岩性
        if chart_type == 'thickness_contour':
            raw_df = self.data_result.get('raw_df')
            if raw_df is not None and 'lithology' in raw_df.columns:
                lithologies = sorted(raw_df['lithology'].unique())
                if lithologies:
                    # 简单选择第一个岩性
                    options['lithology'] = lithologies[0]
                    self.log(f"选择岩性: {lithologies[0]}")

        self.log(f"\n开始生成 {chart_type} 图表...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.generate_btn.setEnabled(False)

        # 启动线程
        self.thread = ChartGenerationThread(chart_type, self.data_result, options)
        self.thread.progress.connect(self.log)
        self.thread.finished.connect(self.on_chart_generated)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_chart_generated(self, result: Dict):
        """图表生成完成"""
        self.current_figure = result['figure']
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        self.log("✓ 图表生成完成！可以导出了")

        # 如果是Plotly图表，尝试在浏览器中显示
        if hasattr(self.current_figure, 'show'):
            try:
                self.current_figure.show()
                self.log("已在浏览器中打开交互式图表")
            except:
                pass

    def on_error(self, error_msg: str):
        """处理错误"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.log(f"✗ 错误: {error_msg}")
        QMessageBox.critical(self, "错误", f"图表生成失败:\n{error_msg[:200]}")

    def export_chart(self):
        """导出图表"""
        if self.current_figure is None:
            QMessageBox.warning(self, "警告", "请先生成图表")
            return

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "chart.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;HTML Files (*.html)"
        )

        if not file_path:
            return

        try:
            from src.visualization import SCIFigureStyle
            import matplotlib.pyplot as plt

            # 确定格式
            ext = os.path.splitext(file_path)[1][1:]

            if hasattr(self.current_figure, 'write_html'):
                # Plotly图表
                if ext == 'html':
                    self.current_figure.write_html(file_path)
                else:
                    self.current_figure.write_image(file_path, scale=3)
            else:
                # Matplotlib图表
                SCIFigureStyle.save_figure(
                    self.current_figure,
                    file_path,
                    formats=[ext],
                    dpi=300,
                    close_after=False
                )

            self.log(f"✓ 图表已导出: {file_path}")

            # 询问是否打开文件夹
            reply = QMessageBox.question(
                self, "导出成功",
                f"图表已保存:\n{file_path}\n\n是否打开所在文件夹?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                folder = os.path.dirname(file_path)
                os.startfile(folder)

        except Exception as e:
            import traceback
            error_msg = f"导出失败: {str(e)}\n{traceback.format_exc()}"
            self.log(f"✗ {error_msg}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")


class MLChartDialog(QDialog):
    """机器学习图表对话框"""

    def __init__(self, parent, data_result: Dict, model, predictor):
        super().__init__(parent)
        self.data_result = data_result
        self.model = model
        self.predictor = predictor

        self.setWindowTitle("机器学习图表生成器")
        self.setMinimumSize(700, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("🤖 机器学习图表生成器")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        info = QLabel("该功能需要模型训练历史数据，暂不可用")
        info.setStyleSheet("color: #999; font-style: italic;")
        layout.addWidget(info)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)


class ResultChartDialog(QDialog):
    """结果分析图表对话框"""

    def __init__(self, parent, data_result: Dict, block_models, XI, YI):
        super().__init__(parent)
        self.data_result = data_result
        self.block_models = block_models
        self.XI = XI
        self.YI = YI

        self.setWindowTitle("结果分析图表生成器")
        self.setMinimumSize(700, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("📈 结果分析图表生成器")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        info = QLabel("该功能将在未来版本中实现")
        info.setStyleSheet("color: #999; font-style: italic;")
        layout.addWidget(info)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
