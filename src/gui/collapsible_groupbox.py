"""
可折叠分组盒子 (Collapsible Group Box)
用于减少滚动，提供更好的控制面板组织
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont
from src.gui.animations import AnimationUtils


class CollapsibleGroupBox(QWidget):
    """可折叠的分组盒子"""

    toggled = pyqtSignal(bool)  # 展开/折叠状态改变信号

    def __init__(self, title: str = "", parent=None, collapsed: bool = False):
        super().__init__(parent)
        self.is_collapsed = collapsed

        self.init_ui(title)

        # 如果初始状态是折叠的，立即隐藏内容
        if self.is_collapsed:
            self.content_widget.setVisible(False)
            self.toggle_button.setText("▶")

    def init_ui(self, title: str):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 标题栏
        header_widget = QFrame()
        header_widget.setObjectName("collapsible_header")
        header_widget.setStyleSheet("""
            QFrame#collapsible_header {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QFrame#collapsible_header:hover {
                background-color: #3a3a4d;
            }
        """)
        header_widget.setCursor(Qt.CursorShape.PointingHandCursor)

        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 5, 8, 5)
        header_layout.setSpacing(8)

        # 折叠/展开按钮
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #cdd6f4;
                font-size: 12px;
                padding: 0px;
            }
            QPushButton:hover {
                color: #89b4fa;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        header_layout.addWidget(self.toggle_button)

        # 标题标签
        self.title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #cdd6f4;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # 使整个header可点击
        header_widget.mousePressEvent = lambda event: self.toggle_collapsed()

        main_layout.addWidget(header_widget)

        # 内容区域
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(10)

        main_layout.addWidget(self.content_widget)

    def toggle_collapsed(self):
        """切换折叠/展开状态"""
        self.is_collapsed = not self.is_collapsed

        if self.is_collapsed:
            self.toggle_button.setText("▶")
            self.content_widget.setVisible(False)
        else:
            self.toggle_button.setText("▼")
            self.content_widget.setVisible(True)
            # Add fade in effect
            AnimationUtils.fade_in(self.content_widget, duration=300)

        self.toggled.emit(not self.is_collapsed)

    def set_collapsed(self, collapsed: bool):
        """设置折叠状态"""
        if self.is_collapsed != collapsed:
            self.toggle_collapsed()

    def add_widget(self, widget: QWidget):
        """添加子控件到内容区域"""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """添加布局到内容区域"""
        self.content_layout.addLayout(layout)

    def add_stretch(self):
        """添加弹性空间"""
        self.content_layout.addStretch()

    def set_title(self, title: str):
        """设置标题"""
        self.title_label.setText(title)
