"""
批量导出对话框
一键生成多个视角的论文图表
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QDialogButtonBox,
    QCheckBox, QComboBox, QSpinBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from typing import List, Dict, Any
import os


class BatchExportWorker(QThread):
    """批量导出工作线程"""
    progress = pyqtSignal(int, str)  # 进度, 消息
    finished = pyqtSignal(list)  # 导出的文件列表
    error = pyqtSignal(str)

    def __init__(self, plotter, camera_presets, export_settings, output_dir):
        super().__init__()
        self.plotter = plotter
        self.camera_presets = camera_presets
        self.export_settings = export_settings
        self.output_dir = output_dir

    def run(self):
        """执行批量导出"""
        try:
            from src.gui.camera_presets import CameraPresetManager
            from src.gui.scientific_annotations import ScientificAnnotations

            exported_files = []
            total = len(self.camera_presets)

            for i, preset_name in enumerate(self.camera_presets):
                self.progress.emit(int((i / total) * 100), f"导出视角: {preset_name}")

                # 应用视角
                bounds = self.plotter.bounds
                success = CameraPresetManager.apply_preset(self.plotter, preset_name, bounds)

                if not success:
                    self.progress.emit(int((i / total) * 100), f"跳过 {preset_name}: 应用视角失败")
                    continue

                self.plotter.render()

                # 构建文件名
                safe_name = preset_name.replace(' ', '_').replace('/', '_')
                filename = f"{safe_name}.{self.export_settings['format']}"
                filepath = os.path.join(self.output_dir, filename)

                # 添加注释
                annotations_added = False
                if self.export_settings.get('add_scale_bar', False):
                    try:
                        ScientificAnnotations.add_scale_bar(
                            self.plotter,
                            position='lower_right',
                            color='black',
                            font_size=14
                        )
                        annotations_added = True
                    except Exception:
                        pass

                if self.export_settings.get('add_north_arrow', False):
                    try:
                        ScientificAnnotations.add_north_arrow(
                            self.plotter,
                            position=(0.9, 0.9),
                            size=50,
                            color='red'
                        )
                        annotations_added = True
                    except Exception:
                        pass

                # 设置窗口大小
                original_size = self.plotter.window_size
                self.plotter.window_size = [
                    self.export_settings['width_px'],
                    self.export_settings['height_px']
                ]
                self.plotter.render()

                # 截图
                self.plotter.screenshot(
                    filepath,
                    transparent_background=self.export_settings.get('transparent_background', False),
                    scale=1
                )

                # 恢复
                self.plotter.window_size = original_size
                if annotations_added:
                    ScientificAnnotations.remove_all_annotations(self.plotter)

                exported_files.append(filepath)
                self.progress.emit(int(((i + 1) / total) * 100), f"完成: {filename}")

            self.finished.emit(exported_files)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class BatchExportDialog(QDialog):
    """批量导出对话框"""

    def __init__(self, parent=None, plotter=None):
        super().__init__(parent)
        self.plotter = plotter
        self.setWindowTitle("批量导出多视角图表")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.worker = None
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)

        # 视角选择
        view_group = QGroupBox("选择导出视角")
        view_layout = QVBoxLayout()

        # 快速选择按钮
        quick_select_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_views)
        deselect_all_btn = QPushButton("取消全选")
        deselect_all_btn.clicked.connect(self.deselect_all_views)
        select_common_btn = QPushButton("常用视角")
        select_common_btn.clicked.connect(self.select_common_views)

        quick_select_layout.addWidget(select_all_btn)
        quick_select_layout.addWidget(deselect_all_btn)
        quick_select_layout.addWidget(select_common_btn)
        quick_select_layout.addStretch()
        view_layout.addLayout(quick_select_layout)

        # 视角列表
        self.view_list = QListWidget()
        self.view_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)

        # 添加所有可用视角
        try:
            from src.gui.camera_presets import CameraPresetManager
            for preset_name in CameraPresetManager.get_preset_names():
                item = QListWidgetItem(preset_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                description = CameraPresetManager.get_preset_description(preset_name)
                item.setToolTip(description)
                self.view_list.addItem(item)
        except Exception as e:
            pass

        view_layout.addWidget(self.view_list)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # 导出设置
        settings_group = QGroupBox("导出设置")
        settings_layout = QVBoxLayout()

        # 格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['png', 'tiff', 'jpeg'])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        settings_layout.addLayout(format_layout)

        # DPI
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(96, 1200)
        self.dpi_spin.setValue(300)
        dpi_layout.addWidget(self.dpi_spin)
        dpi_layout.addStretch()
        settings_layout.addLayout(dpi_layout)

        # 尺寸 (mm)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("宽度(mm):"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(50, 500)
        self.width_spin.setValue(183)
        size_layout.addWidget(self.width_spin)

        size_layout.addWidget(QLabel("高度(mm):"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(50, 500)
        self.height_spin.setValue(137)
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch()
        settings_layout.addLayout(size_layout)

        # 附加选项
        self.add_scale_bar_cb = QCheckBox("添加比例尺")
        self.add_scale_bar_cb.setChecked(True)
        settings_layout.addWidget(self.add_scale_bar_cb)

        self.add_north_arrow_cb = QCheckBox("添加指北针")
        self.add_north_arrow_cb.setChecked(True)
        settings_layout.addWidget(self.add_north_arrow_cb)

        self.transparent_bg_cb = QCheckBox("透明背景")
        settings_layout.addWidget(self.transparent_bg_cb)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_label = QLabel("未选择")
        self.output_dir_label.setStyleSheet("color: gray;")
        output_layout.addWidget(self.output_dir_label, 1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)
        layout.addLayout(output_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 按钮
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始导出")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_export)
        button_layout.addWidget(self.start_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # 默认选择常用视角
        self.select_common_views()

    def select_all_views(self):
        """全选视角"""
        for i in range(self.view_list.count()):
            self.view_list.item(i).setCheckState(Qt.CheckState.Checked)

    def deselect_all_views(self):
        """取消全选"""
        for i in range(self.view_list.count()):
            self.view_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def select_common_views(self):
        """选择常用视角（论文标准）"""
        common_views = ["立体全景", "纯俯视图", "正北剖面", "正东剖面"]
        for i in range(self.view_list.count()):
            item = self.view_list.item(i)
            if item.text() in common_views:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

    def browse_output_dir(self):
        """浏览输出目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.output_dir_label.setText(directory)
            self.output_dir_label.setStyleSheet("color: black;")

    def start_export(self):
        """开始批量导出"""
        # 检查是否选择了视角
        selected_views = []
        for i in range(self.view_list.count()):
            item = self.view_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_views.append(item.text())

        if not selected_views:
            QMessageBox.warning(self, "警告", "请至少选择一个视角！")
            return

        # 检查输出目录
        output_dir = self.output_dir_label.text()
        if output_dir == "未选择" or not os.path.exists(output_dir):
            QMessageBox.warning(self, "警告", "请选择有效的输出目录！")
            return

        # 准备导出设置
        dpi = self.dpi_spin.value()
        width_mm = self.width_spin.value()
        height_mm = self.height_spin.value()

        export_settings = {
            'format': self.format_combo.currentText(),
            'dpi': dpi,
            'width_px': int(width_mm / 25.4 * dpi),
            'height_px': int(height_mm / 25.4 * dpi),
            'add_scale_bar': self.add_scale_bar_cb.isChecked(),
            'add_north_arrow': self.add_north_arrow_cb.isChecked(),
            'transparent_background': self.transparent_bg_cb.isChecked()
        }

        # 禁用开始按钮
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append(f"开始批量导出 {len(selected_views)} 个视角...")

        # 创建工作线程
        self.worker = BatchExportWorker(
            self.plotter, selected_views, export_settings, output_dir
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, percent: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(percent)
        self.log_text.append(message)

    def on_finished(self, files: List[str]):
        """导出完成"""
        self.progress_bar.setValue(100)
        self.log_text.append(f"\n✓ 批量导出完成！共导出 {len(files)} 个文件。")
        self.start_btn.setEnabled(True)

        # 询问是否打开文件夹
        reply = QMessageBox.question(
            self, "导出成功",
            f"已成功导出 {len(files)} 个图像！\n\n是否打开输出文件夹?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                import os
                os.startfile(self.output_dir_label.text())
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开文件夹: {e}")

    def on_error(self, error_msg: str):
        """导出错误"""
        self.log_text.append(f"\n✗ 导出失败: {error_msg}")
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", f"批量导出失败:\n{error_msg}")
