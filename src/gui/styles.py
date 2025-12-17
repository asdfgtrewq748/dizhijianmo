MODERN_STYLE = """
/* 全局样式 */
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}

/* 滚动区域背景 */
QScrollArea {
    background-color: #1e1e2e;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: #1e1e2e;
}

/* 分组框 */
QGroupBox {
    border: 2px solid #313244;
    border-radius: 8px;
    margin-top: 24px;
    padding-top: 12px;
    background-color: #252635;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 6px 12px;
    background-color: #313244;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: #89b4fa;
    font-size: 15px;
}

/* 按钮通用 */
QPushButton {
    background-color: #45475a;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    color: #ffffff;
    font-weight: bold;
    font-size: 14px;
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

/* 主要操作按钮 (蓝色) */
QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton#primary:hover {
    background-color: #b4befe;
}
QPushButton#primary:pressed {
    background-color: #74c7ec;
}

/* 成功/导出按钮 (绿色) */
QPushButton#success {
    background-color: #a6e3a1;
    color: #1e1e2e;
}
QPushButton#success:hover {
    background-color: #94e2d5;
}

/* 输入控件 */
QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
    color: #cdd6f4;
    selection-background-color: #585b70;
    min-height: 20px;
}
QComboBox::drop-down {
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid #cdd6f4;
    margin-right: 8px;
}

/* 滚动条 */
QScrollBar:vertical {
    border: none;
    background: #1e1e2e;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 进度条 */
QProgressBar {
    border: none;
    background-color: #313244;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    min-height: 20px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 4px;
}

/* 分割器 */
QSplitter::handle {
    background-color: #45475a;
    width: 4px;
}

/* 标签 */
QLabel {
    color: #cdd6f4;
    padding: 2px;
}
QLabel#header {
    color: #89b4fa;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 0;
}

/* 复选框 */
QCheckBox {
    spacing: 10px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 1px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

/* 滑块 */
QSlider::groove:horizontal {
    border: 1px solid #45475a;
    height: 8px;
    background: #313244;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: 1px solid #89b4fa;
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
}
"""
