"""
专业级导出对话框
用于科研论文图表导出，支持高DPI和多种尺寸选项
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QCheckBox, QPushButton, QDialogButtonBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from typing import Dict, Any


class PublicationExportDialog(QDialog):
    """论文级图表导出对话框"""

    # 期刊标准尺寸（毫米）
    JOURNAL_SIZES = {
        "单栏 (89mm)": (89, 89),
        "1.5栏 (140mm)": (140, 105),
        "双栏 (183mm)": (183, 137),
        "A4纸 (210mm)": (210, 297),
        "自定义": None
    }

    # DPI预设
    DPI_PRESETS = {
        "屏幕显示 (96 DPI)": 96,
        "标准打印 (150 DPI)": 150,
        "高质量打印 (300 DPI)": 300,
        "出版印刷 (600 DPI)": 600,
        "超高分辨率 (1200 DPI)": 1200,
        "自定义": None
    }

    def __init__(self, parent=None, default_format='png'):
        super().__init__(parent)
        self.setWindowTitle("专业导出设置")
        self.setMinimumWidth(500)

        self.init_ui()
        self.load_defaults(default_format)

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 格式选择
        format_group = QGroupBox("导出格式")
        format_layout = QVBoxLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "PNG - 便携式网络图形 (推荐)",
            "TIFF - 标记图像文件格式 (无损)",
            "PDF - 便携式文档格式 (矢量)",
            "SVG - 可缩放矢量图形 (矢量)",
            "EPS - 封装PostScript (矢量)",
            "JPEG - 联合图像专家组 (有损)"
        ])
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)

        format_layout.addWidget(QLabel("文件格式:"))
        format_layout.addWidget(self.format_combo)

        self.transparent_bg_cb = QCheckBox("透明背景")
        self.transparent_bg_cb.setToolTip("PNG/TIFF格式支持透明背景")
        format_layout.addWidget(self.transparent_bg_cb)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # 尺寸设置
        size_group = QGroupBox("图像尺寸")
        size_layout = QVBoxLayout()

        size_layout.addWidget(QLabel("期刊标准尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(self.JOURNAL_SIZES.keys()))
        self.size_combo.currentTextChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_combo)

        # 自定义尺寸输入
        custom_size_layout = QHBoxLayout()
        custom_size_layout.addWidget(QLabel("宽度 (mm):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 500)
        self.width_spin.setValue(183)
        self.width_spin.setSingleStep(1)
        custom_size_layout.addWidget(self.width_spin)

        custom_size_layout.addWidget(QLabel("高度 (mm):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(10, 500)
        self.height_spin.setValue(137)
        self.height_spin.setSingleStep(1)
        custom_size_layout.addWidget(self.height_spin)

        size_layout.addLayout(custom_size_layout)

        # 宽高比锁定
        self.lock_aspect_cb = QCheckBox("锁定宽高比")
        self.lock_aspect_cb.setChecked(True)
        size_layout.addWidget(self.lock_aspect_cb)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # DPI设置
        dpi_group = QGroupBox("分辨率 (DPI)")
        dpi_layout = QVBoxLayout()

        dpi_layout.addWidget(QLabel("DPI预设:"))
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(list(self.DPI_PRESETS.keys()))
        self.dpi_combo.setCurrentText("高质量打印 (300 DPI)")
        self.dpi_combo.currentTextChanged.connect(self.on_dpi_changed)
        dpi_layout.addWidget(self.dpi_combo)

        # 自定义DPI
        custom_dpi_layout = QHBoxLayout()
        custom_dpi_layout.addWidget(QLabel("自定义DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 2400)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        custom_dpi_layout.addWidget(self.dpi_spin)
        custom_dpi_layout.addStretch()
        dpi_layout.addLayout(custom_dpi_layout)

        # 像素尺寸预览
        self.pixel_size_label = QLabel()
        self.update_pixel_size_preview()
        dpi_layout.addWidget(self.pixel_size_label)

        # 连接信号以更新预览
        self.width_spin.valueChanged.connect(self.update_pixel_size_preview)
        self.height_spin.valueChanged.connect(self.update_pixel_size_preview)
        self.dpi_spin.valueChanged.connect(self.update_pixel_size_preview)

        dpi_group.setLayout(dpi_layout)
        layout.addWidget(dpi_group)

        # 额外选项
        options_group = QGroupBox("附加选项")
        options_layout = QVBoxLayout()

        self.add_scale_bar_cb = QCheckBox("添加比例尺")
        self.add_scale_bar_cb.setChecked(True)
        options_layout.addWidget(self.add_scale_bar_cb)

        self.add_north_arrow_cb = QCheckBox("添加指北针")
        self.add_north_arrow_cb.setChecked(True)
        options_layout.addWidget(self.add_north_arrow_cb)

        self.add_legend_cb = QCheckBox("包含图例")
        self.add_legend_cb.setChecked(True)
        options_layout.addWidget(self.add_legend_cb)

        self.add_axes_cb = QCheckBox("显示坐标轴")
        self.add_axes_cb.setChecked(False)
        options_layout.addWidget(self.add_axes_cb)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 应用样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def load_defaults(self, format_name: str):
        """加载默认设置"""
        format_map = {
            'png': 0,
            'tiff': 1,
            'pdf': 2,
            'svg': 3,
            'eps': 4,
            'jpeg': 5
        }
        if format_name.lower() in format_map:
            self.format_combo.setCurrentIndex(format_map[format_name.lower()])

    def on_format_changed(self, index):
        """格式改变时的处理"""
        format_text = self.format_combo.currentText()

        # 只有PNG和TIFF支持透明背景
        if 'PNG' in format_text or 'TIFF' in format_text:
            self.transparent_bg_cb.setEnabled(True)
        else:
            self.transparent_bg_cb.setEnabled(False)
            self.transparent_bg_cb.setChecked(False)

        # 矢量格式不需要DPI
        if 'PDF' in format_text or 'SVG' in format_text or 'EPS' in format_text:
            self.dpi_combo.setEnabled(False)
            self.dpi_spin.setEnabled(False)
        else:
            self.dpi_combo.setEnabled(True)
            self.dpi_spin.setEnabled(True)

    def on_size_changed(self, text):
        """尺寸预设改变"""
        if text == "自定义":
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)
        else:
            size = self.JOURNAL_SIZES.get(text)
            if size:
                self.width_spin.setValue(size[0])
                self.height_spin.setValue(size[1])
                self.width_spin.setEnabled(False)
                self.height_spin.setEnabled(False)

    def on_dpi_changed(self, text):
        """DPI预设改变"""
        if text == "自定义":
            self.dpi_spin.setEnabled(True)
        else:
            dpi = self.DPI_PRESETS.get(text)
            if dpi:
                self.dpi_spin.setValue(dpi)
                self.dpi_spin.setEnabled(False)

    def update_pixel_size_preview(self):
        """更新像素尺寸预览"""
        width_mm = self.width_spin.value()
        height_mm = self.height_spin.value()
        dpi = self.dpi_spin.value()

        # 转换为像素 (1英寸 = 25.4mm)
        width_px = int(width_mm / 25.4 * dpi)
        height_px = int(height_mm / 25.4 * dpi)

        # 估算文件大小（未压缩）
        file_size_mb = (width_px * height_px * 4) / (1024 * 1024)  # RGBA

        self.pixel_size_label.setText(
            f"输出像素尺寸: {width_px} × {height_px} px\n"
            f"预估文件大小: {file_size_mb:.1f} MB (未压缩)"
        )
        self.pixel_size_label.setStyleSheet("color: #7f849c; font-size: 11px;")

    def get_export_settings(self) -> Dict[str, Any]:
        """获取导出设置"""
        format_text = self.format_combo.currentText()
        format_ext = format_text.split(' - ')[0].lower()

        return {
            'format': format_ext,
            'width_mm': self.width_spin.value(),
            'height_mm': self.height_spin.value(),
            'dpi': self.dpi_spin.value(),
            'transparent_background': self.transparent_bg_cb.isChecked(),
            'add_scale_bar': self.add_scale_bar_cb.isChecked(),
            'add_north_arrow': self.add_north_arrow_cb.isChecked(),
            'add_legend': self.add_legend_cb.isChecked(),
            'add_axes': self.add_axes_cb.isChecked(),
            # 计算像素尺寸
            'width_px': int(self.width_spin.value() / 25.4 * self.dpi_spin.value()),
            'height_px': int(self.height_spin.value() / 25.4 * self.dpi_spin.value())
        }
