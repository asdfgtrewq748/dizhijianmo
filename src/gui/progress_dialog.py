"""
现代化进度对话框 - 带动画和百分比显示

用于显示长时间运行任务（训练、建模、导出）的进度
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont
import time


class ModernProgressDialog(QDialog):
    """
    现代化进度对话框

    特性:
    - 百分比进度条
    - 动态消息更新
    - 脉冲动画（不确定进度时）
    - 时间估计
    - 取消按钮
    - Catppuccin Mocha 主题风格
    """

    cancel_requested = pyqtSignal()  # 取消信号

    def __init__(self, parent=None, title="处理中...", message="正在初始化...", cancelable=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)

        # 禁用关闭按钮
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.CustomizeWindowHint
        )

        self._cancelable = cancelable
        self._start_time = time.time()
        self._last_progress = 0
        self._last_progress_time = self._start_time

        self._setup_ui(message)
        self._apply_theme()

        # 启动时间更新定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed_time)
        self._timer.start(1000)  # 每秒更新一次

    def _setup_ui(self, message):
        """设置UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        # 标题标签
        self.title_label = QLabel(message)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(30)
        layout.addWidget(self.progress_bar)

        # 详细信息标签
        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_font = QFont()
        detail_font.setPointSize(9)
        self.detail_label.setFont(detail_font)
        self.detail_label.setStyleSheet("color: #a6adc8;")  # 稍微浅一点的颜色
        layout.addWidget(self.detail_label)

        # 时间信息标签
        self.time_label = QLabel("已用时间: 0秒")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_font = QFont()
        time_font.setPointSize(8)
        self.time_label.setFont(time_font)
        self.time_label.setStyleSheet("color: #585b70;")
        layout.addWidget(self.time_label)

        # 弹性空间
        layout.addStretch()

        # 取消按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setMinimumWidth(100)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        if not self._cancelable:
            self.cancel_button.hide()
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _apply_theme(self):
        """应用 Catppuccin Mocha 主题"""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }

            QLabel {
                color: #cdd6f4;
            }

            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 8px;
                background-color: #313244;
                text-align: center;
                color: #cdd6f4;
                font-weight: bold;
                font-size: 11pt;
            }

            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #89b4fa,
                    stop: 0.5 #74c7ec,
                    stop: 1 #94e2d5
                );
                border-radius: 6px;
            }

            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
            }

            QPushButton:hover {
                background-color: #585b70;
            }

            QPushButton:pressed {
                background-color: #313244;
            }

            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
        """)

    def _on_cancel_clicked(self):
        """取消按钮点击"""
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("正在取消...")
        self.set_message("正在取消...")
        self.cancel_requested.emit()

    def _update_elapsed_time(self):
        """更新已用时间和预计剩余时间"""
        elapsed = time.time() - self._start_time
        elapsed_str = self._format_time(elapsed)

        # 计算预计剩余时间
        current_progress = self.progress_bar.value()
        if current_progress > 0 and current_progress < 100:
            # 基于当前进度估算总时间
            estimated_total = elapsed / (current_progress / 100.0)
            remaining = estimated_total - elapsed
            remaining_str = self._format_time(remaining)
            time_text = f"已用时间: {elapsed_str}  |  预计剩余: {remaining_str}"
        else:
            time_text = f"已用时间: {elapsed_str}"

        self.time_label.setText(time_text)

    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分"

    def set_progress(self, value: int):
        """
        设置进度值

        Args:
            value: 0-100 之间的整数
        """
        if value < 0:
            value = 0
        if value > 100:
            value = 100

        self._last_progress = value
        self._last_progress_time = time.time()
        self.progress_bar.setValue(value)

    def set_message(self, message: str):
        """
        设置主消息

        Args:
            message: 消息文本
        """
        self.title_label.setText(message)

    def set_detail(self, detail: str):
        """
        设置详细信息

        Args:
            detail: 详细信息文本
        """
        self.detail_label.setText(detail)

    def set_indeterminate(self, is_indeterminate: bool = True):
        """
        设置为不确定模式（脉冲动画）

        Args:
            is_indeterminate: True 为不确定模式，False 为正常模式
        """
        if is_indeterminate:
            self.progress_bar.setMaximum(0)  # 不确定模式
        else:
            self.progress_bar.setMaximum(100)

    def enable_cancel(self, enable: bool = True):
        """
        启用/禁用取消按钮

        Args:
            enable: True 显示取消按钮，False 隐藏
        """
        self._cancelable = enable
        if enable:
            self.cancel_button.show()
            self.cancel_button.setEnabled(True)
            self.cancel_button.setText("取消")
        else:
            self.cancel_button.hide()

    def auto_close_on_complete(self):
        """完成时自动关闭（延迟500ms以显示100%）"""
        self._timer.stop()  # 停止时间更新
        QTimer.singleShot(500, self.accept)

    def closeEvent(self, event):
        """关闭事件 - 停止定时器"""
        self._timer.stop()
        super().closeEvent(event)


# 便捷函数
def create_progress_dialog(parent, title, message, cancelable=False):
    """
    创建进度对话框的便捷函数

    Args:
        parent: 父窗口
        title: 对话框标题
        message: 初始消息
        cancelable: 是否可取消

    Returns:
        ModernProgressDialog 实例
    """
    dialog = ModernProgressDialog(parent, title, message, cancelable)
    return dialog
