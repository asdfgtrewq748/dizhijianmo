
MODERN_STYLE = """
/* ==================================================================================
   高级材质 UI 样式表 (Premium Material Theme)
   基于 Catppuccin Mocha 配色，增加了渐变、阴影和深度感
   ================================================================================= */

/* 全局基础 */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e1e2e, stop:1 #181825);
    color: #cdd6f4;
}

QWidget {
    background-color: transparent;
    color: #cdd6f4;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}

/* 滚动区域 */
QScrollArea {
    background-color: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

/* ==================================================================================
   容器与分组
   ================================================================================== */

/* 分组框 - 卡片式设计 */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 12px;
    margin-top: 32px; /* 为标题留出空间 */
    background-color: rgba(30, 30, 46, 0.7); /* 半透明背景 */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 8px 16px;
    margin-left: 10px;
    
    background-color: #313244;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    
    color: #89b4fa;
    font-weight: bold;
    font-size: 14px;
}

/* 可折叠头部样式 (配合 CollapsibleGroupBox) */
QFrame#collapsible_header {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #313244, stop:1 #262736);
    border: 1px solid #45475a;
    border-radius: 8px;
}
QFrame#collapsible_header:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a3a4d, stop:1 #313244);
    border-color: #585b70;
}

/* ==================================================================================
   按钮样式
   ================================================================================== */

/* 通用按钮 - 玻璃质感 */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #45475a, stop:1 #313244);
    border: 1px solid #585b70;
    border-radius: 8px;
    padding: 8px 16px;
    color: #ffffff;
    font-weight: 600;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #585b70, stop:1 #45475a);
    border-color: #7f849c;
}

QPushButton:pressed {
    background-color: #1e1e2e;
    border-color: #313244;
    padding-top: 10px; /* 按下位移效果 */
    padding-bottom: 6px;
}

QPushButton:disabled {
    background-color: #313244;
    border-color: #45475a;
    color: #6c7086;
}

/* 主要操作按钮 (蓝色高亮) */
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #89b4fa, stop:1 #74c7ec);
    border: 1px solid #89b4fa;
    color: #1e1e2e;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #b4befe, stop:1 #89b4fa);
    border-color: #b4befe;
}
QPushButton#primary:pressed {
    background: #74c7ec;
}

/* 成功/导出按钮 (绿色高亮) */
QPushButton#success {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a6e3a1, stop:1 #94e2d5);
    border: 1px solid #a6e3a1;
    color: #1e1e2e;
}
QPushButton#success:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c6eac3, stop:1 #a6e3a1);
}

/* ==================================================================================
   输入控件
   ================================================================================== */

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px;
    color: #cdd6f4;
    selection-background-color: #585b70;
    selection-color: #ffffff;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QListWidget:focus {
    border: 1px solid #89b4fa;
    background-color: #1e1e2e;
}

/* 下拉框美化 */
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #89b4fa;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    selection-background-color: #313244;
    outline: none;
}

/* 列表控件 */
QListWidget::item {
    padding: 8px;
    border-radius: 4px;
    margin: 2px;
}
QListWidget::item:hover {
    background-color: #313244;
}
QListWidget::item:selected {
    background-color: #45475a;
    border: 1px solid #89b4fa;
}

/* ==================================================================================
   其他控件
   ================================================================================== */

/* 滚动条 - 极简风格 */
QScrollBar:vertical {
    border: none;
    background: #181825;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 进度条 - 霓虹效果 */
QProgressBar {
    border: 1px solid #45475a;
    background-color: #181825;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #89b4fa, stop:1 #b4befe);
    border-radius: 5px;
}

/* 分割器 */
QSplitter::handle {
    background-color: #313244;
    width: 2px;
    margin: 0 2px;
}
QSplitter::handle:hover {
    background-color: #89b4fa;
}

/* 标签 */
QLabel {
    color: #cdd6f4;
}
QLabel#header {
    color: #89b4fa;
    font-size: 20px;
    font-weight: 800;
    padding: 12px 0;
    font-family: "Segoe UI", sans-serif;
}

/* 复选框 */
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #585b70;
    background-color: #181825;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
    image: url(none); /* 可以添加自定义勾选图标 */
}
QCheckBox::indicator:checked:hover {
    background-color: #b4befe;
}

/* 滑块 - 现代风格 */
QSlider::groove:horizontal {
    border: 1px solid #45475a;
    height: 6px;
    background: #181825;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: 2px solid #1e1e2e;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #b4befe;
    transform: scale(1.2);
}

/* 菜单栏 */
QMenuBar {
    background-color: #1e1e2e;
    border-bottom: 1px solid #313244;
}
QMenuBar::item {
    padding: 8px 12px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #313244;
    border-radius: 4px;
}
QMenu {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}
"""
