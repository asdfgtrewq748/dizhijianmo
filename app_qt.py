"""
GNN地质建模系统 - PyQt6高性能增强版

特性:
- PyQt6 原生界面
- PyVista GPU加速3D渲染
- 多线程数据处理
- CUDA加速训练
- 实时进度反馈
- 完整渲染控制（层选择、模式切换、透明度、侧面、钻孔）
- 多格式导出（PNG/HTML/OBJ/STL/VTK）

性能提升:
- 建模速度: 5-10倍
- 渲染帧率: 60+ FPS
- GPU利用率: 70-90%

启动: python app_qt.py
版本: v2.0 增强版
"""

import sys
import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QProgressBar, QTabWidget, QCheckBox,
    QSplitter, QSlider, QListWidget, QListWidgetItem, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QMenuBar, QMenu, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QFont, QTextCursor, QAction, QCloseEvent

# PyVista + Qt集成
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: pyvistaqt not installed. 3D rendering will be disabled.")
    print("Install with: pip install pyvistaqt")

# 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.thickness_data_loader import ThicknessDataProcessor
from src.gnn_thickness_modeling import (
    GNNThicknessPredictor, GeologicalModelBuilder
)
from src.thickness_trainer import create_trainer, get_optimized_config_for_small_dataset
from src.thickness_predictor_v2 import (
    PerLayerThicknessPredictor, HybridThicknessPredictor, evaluate_predictor
)

# Refactored GUI modules
from src.gui.workers import (
    DataLoaderThread, TrainingThread,
    TraditionalPredictorThread, ModelingThread
)
from src.gui.dialogs import BoreholeInfoDialog
from src.gui.progress_dialog import ModernProgressDialog
from src.gui.styles import MODERN_STYLE
from src.gui.utils import setup_logging, global_exception_hook

if PYVISTA_AVAILABLE:
    from src.pyvista_renderer import GeologicalModelRenderer, RockMaterial, TextureGenerator

# FLAC3D导出器
try:
    from src.exporters.flac3d_enhanced_exporter import EnhancedFLAC3DExporter
    from src.exporters.flac3d_compact_exporter import CompactFLAC3DExporter
    from src.exporters.f3grid_exporter_v2 import F3GridExporterV2
    from src.exporters.fpn_exporter import FPNExporter
    FLAC3D_EXPORTER_AVAILABLE = True
    F3GRID_V2_AVAILABLE = True
    FPN_EXPORTER_AVAILABLE = True
except ImportError as e:
    FLAC3D_EXPORTER_AVAILABLE = False
    F3GRID_V2_AVAILABLE = False
    FPN_EXPORTER_AVAILABLE = False
    print(f"Warning: FLAC3D exporter not available: {e}")


# =============================================================================
# 工作线程 - 多线程处理，避免UI阻塞
# =============================================================================

# Threads have been moved to src/gui/workers.py



# =============================================================================
# 钻孔信息对话框
# =============================================================================

# Moved to src/gui/dialogs.py


# =============================================================================
# 主窗口
# =============================================================================

class GeologicalModelingApp(QMainWindow):
    """地质建模主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GNN地质建模系统 - PyQt6高性能增强版 v2.0")
        self.setGeometry(100, 100, 1600, 900)

        self.data_result = None
        self.model = None
        self.predictor = None
        self.block_models = None
        self.XI = None
        self.YI = None
        self.use_traditional = False
        
        # 渲染缓存
        self.cached_meshes = {}
        self.cached_textures = {} # 纹理缓存
        self.cached_sides_state = None
        self.mesh_cache = {}  # 按是否生成侧面缓存不同的网格
        self.merged_mesh_cache = None  # 剖面切割用的合并网格缓存
        self.is_rendering = False
        
        # 渲染状态跟踪
        self.last_render_params = {}
        self.actors_map = {}
        
        # 实时更新状态
        self.last_base_level = 0.0
        self.resolution_timer = QTimer()
        self.resolution_timer.setSingleShot(True)
        self.resolution_timer.setInterval(1000) # 1秒延迟
        self.resolution_timer.timeout.connect(self.build_3d_model)
        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(200)  # 渲染防抖
        self.render_timer.timeout.connect(self.render_3d_model)

        if getattr(sys, 'frozen', False):
            self.project_root = Path(sys.executable).parent
        else:
            self.project_root = Path(__file__).parent
            
        self.data_dir = self.project_root / 'data'
        self.texture_dir = self.project_root / 'textures'

        self.init_ui()
        self.setup_logging()
        self.check_gpu()
        self.load_settings()

    def setup_logging(self):
        """Setup logging system"""
        self.log_handler = setup_logging()
        self.log_handler.new_record.connect(self.append_log)

    def apply_modern_style(self):
        """应用现代深色主题样式"""
        self.setStyleSheet(MODERN_STYLE)

    def request_render(self, delay_ms: int = 200):
        """防抖触发渲染，避免频繁重绘卡顿"""
        if not PYVISTA_AVAILABLE or self.plotter is None:
            return
        self.render_timer.setInterval(delay_ms)
        self.render_timer.start()

    def init_ui(self):
        """初始化用户界面"""
        self.apply_modern_style()
        self.create_menu_bar()
        
        # 启用拖拽
        self.setAcceptDrops(True)
        
        self.log_text = None
        self.stats_text = None
        self.progress_bar = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)

        center_panel = self.create_render_panel()
        splitter.addWidget(center_panel)
        
        right_panel = self.create_info_panel()
        splitter.addWidget(right_panel)

        # 设置初始比例和伸缩因子
        splitter.setSizes([320, 960, 320])
        splitter.setStretchFactor(0, 0) # 左侧不自动伸缩
        splitter.setStretchFactor(1, 1) # 中间自动伸缩
        splitter.setStretchFactor(2, 0) # 右侧不自动伸缩
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)
        splitter.setHandleWidth(4)

        main_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪 | GPU: 检测中...")

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        open_action = QAction('打开项目(&P)...', self)
        open_action.setShortcut('Ctrl+Shift+O')
        open_action.triggered.connect(self.load_project)
        file_menu.addAction(open_action)
        
        save_action = QAction('保存项目(&S)...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()

        load_data_action = QAction('加载数据(&L)', self)
        load_data_action.setShortcut('Ctrl+O')
        load_data_action.triggered.connect(self.load_data)
        file_menu.addAction(load_data_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        
        refresh_action = QAction('刷新渲染(&R)', self)
        refresh_action.setShortcut('Ctrl+R')
        refresh_action.triggered.connect(self.refresh_render)
        view_menu.addAction(refresh_action)

    def load_settings(self):
        """加载用户配置"""
        settings = QSettings("GNN_GeoMod", "App")
        
        # 恢复窗口大小和位置
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        # 恢复上次的数据目录
        last_dir = settings.value("last_data_dir")
        if last_dir and os.path.exists(last_dir):
            self.data_dir = Path(last_dir)
            
        # 恢复参数
        if hasattr(self, 'k_neighbors_spin'):
            self.k_neighbors_spin.setValue(int(settings.value("k_neighbors", 10)))
            
        if hasattr(self, 'resolution_spin'):
            self.resolution_spin.setValue(int(settings.value("resolution", 50)))

    def save_settings(self):
        """保存用户配置"""
        settings = QSettings("GNN_GeoMod", "App")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("last_data_dir", str(self.data_dir))
        
        if hasattr(self, 'k_neighbors_spin'):
            settings.setValue("k_neighbors", self.k_neighbors_spin.value())
            
        if hasattr(self, 'resolution_spin'):
            settings.setValue("resolution", self.resolution_spin.value())

    def closeEvent(self, event: QCloseEvent):
        """窗口关闭事件"""
        self.save_settings()
        event.accept()

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """拖拽释放事件"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith('.json'):
                self.load_project_file(f)
                break # 只加载第一个项目文件
            elif f.lower().endswith('.csv'):
                # 如果是CSV，询问是否作为数据目录加载
                reply = QMessageBox.question(
                    self, "加载数据", 
                    f"检测到CSV文件: {os.path.basename(f)}\n是否将所在目录设置为数据源并加载?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.data_dir = Path(os.path.dirname(f))
                    self.load_data()
                break

    def save_project(self):
        """保存项目状态"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.save_project_file(file_path)

    def save_project_file(self, file_path):
        """保存项目到文件"""
        project_data = {
            "version": "2.0",
            "data_dir": str(self.data_dir),
            "params": {
                "merge_coal": self.merge_coal_cb.isChecked(),
                "layer_method": self.layer_method_combo.currentText(),
                "k_neighbors": self.k_neighbors_spin.value(),
                "min_occurrence": self.min_occurrence_spin.value(),
                "resolution": self.resolution_spin.value(),
                "base_level": self.base_level_spin.value(),
                "use_traditional": self.traditional_radio.isChecked(),
                "interp_method": self.interp_method_combo.currentText()
            }
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=4, ensure_ascii=False)
            self.log(f"✓ 项目已保存: {file_path}")
            self.statusBar().showMessage(f"项目已保存: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def load_project(self):
        """加载项目状态"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.load_project_file(file_path)

    def load_project_file(self, file_path):
        """从文件加载项目"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
                
            # 恢复参数
            params = project_data.get("params", {})
            self.merge_coal_cb.setChecked(params.get("merge_coal", True))
            self.layer_method_combo.setCurrentText(params.get("layer_method", "position_based"))
            self.k_neighbors_spin.setValue(params.get("k_neighbors", 10))
            self.min_occurrence_spin.setValue(params.get("min_occurrence", 0.05))
            self.resolution_spin.setValue(params.get("resolution", 50))
            self.base_level_spin.setValue(params.get("base_level", 0.0))
            
            if params.get("use_traditional", True):
                self.traditional_radio.setChecked(True)
            else:
                self.gnn_radio.setChecked(True)
                
            self.interp_method_combo.setCurrentText(params.get("interp_method", "idw"))
            
            # 恢复数据目录
            data_dir = project_data.get("data_dir")
            if data_dir and os.path.exists(data_dir):
                self.data_dir = Path(data_dir)
                reply = QMessageBox.question(
                    self, "加载数据", 
                    f"项目包含数据目录: {data_dir}\n是否立即加载数据?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.load_data()
            
            self.log(f"✓ 项目已加载: {file_path}")
            self.statusBar().showMessage(f"项目已加载: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def create_control_panel(self) -> QWidget:
        """创建左侧控制面板"""
        # 创建滚动区域容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        title = QLabel("⚙️ 参数设置")
        title.setObjectName("header")
        layout.addWidget(title)

        # 数据配置
        data_group = QGroupBox("数据配置")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)

        self.merge_coal_cb = QCheckBox("合并煤层")
        self.merge_coal_cb.setToolTip("是否将所有煤层合并为一个'Coal'层，以简化模型。")
        data_layout.addWidget(self.merge_coal_cb)

        data_layout.addWidget(QLabel("层序推断方法:"))
        self.layer_method_combo = QComboBox()
        self.layer_method_combo.addItems(['position_based', 'simple', 'marker_based'])
        self.layer_method_combo.setToolTip("推断地层层序的方法：\n- position_based: 基于深度位置\n- simple: 简单统计\n- marker_based: 基于标志层")
        data_layout.addWidget(self.layer_method_combo)

        data_layout.addWidget(QLabel("K邻居数:"))
        self.k_neighbors_spin = QSpinBox()
        self.k_neighbors_spin.setRange(4, 20)
        self.k_neighbors_spin.setValue(10)
        self.k_neighbors_spin.setToolTip("构建图网络时的邻居节点数量 (K)。\n值越大，连接越稠密，计算越慢但可能更平滑。")
        data_layout.addWidget(self.k_neighbors_spin)

        data_layout.addWidget(QLabel("最小出现率:"))
        self.min_occurrence_spin = QDoubleSpinBox()
        self.min_occurrence_spin.setRange(0.0, 0.5)
        self.min_occurrence_spin.setValue(0.05)
        self.min_occurrence_spin.setSingleStep(0.05)
        self.min_occurrence_spin.setToolTip("地层在所有钻孔中出现的最小比例。\n低于此比例的地层将被忽略。")
        data_layout.addWidget(self.min_occurrence_spin)

        self.load_btn = QPushButton("🔄 加载数据")
        self.load_btn.setObjectName("primary")
        self.load_btn.clicked.connect(self.load_data)
        data_layout.addWidget(self.load_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # 预测方法
        method_group = QGroupBox("🔧 预测方法")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(10)

        self.traditional_radio = QCheckBox("传统方法 (IDW/Kriging)")
        self.traditional_radio.setChecked(True)
        self.traditional_radio.stateChanged.connect(self.on_method_changed)
        self.traditional_radio.setToolTip("使用反距离加权(IDW)或克里金(Kriging)插值。")
        method_layout.addWidget(self.traditional_radio)

        self.traditional_params = QWidget()
        trad_layout = QVBoxLayout(self.traditional_params)
        trad_layout.setContentsMargins(0, 0, 0, 0)
        trad_layout.addWidget(QLabel("插值方法:"))
        self.interp_method_combo = QComboBox()
        self.interp_method_combo.addItems(['idw', 'kriging', 'hybrid'])
        trad_layout.addWidget(self.interp_method_combo)
        method_layout.addWidget(self.traditional_params)

        self.gnn_radio = QCheckBox("GNN深度学习")
        self.gnn_radio.stateChanged.connect(self.on_method_changed)
        self.gnn_radio.setToolTip("使用图神经网络(GNN)进行深度学习预测。")
        method_layout.addWidget(self.gnn_radio)

        self.gnn_params = QWidget()
        gnn_layout = QVBoxLayout(self.gnn_params)
        gnn_layout.setContentsMargins(0, 0, 0, 0)

        self.auto_config_cb = QCheckBox("自动优化配置")
        self.auto_config_cb.setChecked(True)
        gnn_layout.addWidget(self.auto_config_cb)

        gnn_layout.addWidget(QLabel("训练轮数:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(100, 500)
        self.epochs_spin.setValue(200)
        gnn_layout.addWidget(self.epochs_spin)

        self.gnn_params.setLayout(gnn_layout)
        self.gnn_params.setVisible(False)
        method_layout.addWidget(self.gnn_params)

        self.train_btn = QPushButton("开始训练")
        self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        method_layout.addWidget(self.train_btn)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # 建模配置
        modeling_group = QGroupBox("建模配置")
        modeling_layout = QVBoxLayout()
        modeling_layout.setSpacing(10)

        modeling_layout.addWidget(QLabel("网格分辨率:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(20, 200)
        self.resolution_spin.setValue(50)
        self.resolution_spin.setToolTip("输出网格的分辨率 (X/Y方向的网格数量)。\n值越大，模型越精细，但内存消耗和计算时间呈平方增长。")
        self.resolution_spin.valueChanged.connect(self.on_resolution_changed)
        modeling_layout.addWidget(self.resolution_spin)

        modeling_layout.addWidget(QLabel("基准面高程(m):"))
        self.base_level_spin = QDoubleSpinBox()
        self.base_level_spin.setValue(0.0)
        self.base_level_spin.valueChanged.connect(self.on_base_level_changed)
        modeling_layout.addWidget(self.base_level_spin)

        self.model_btn = QPushButton("构建三维模型")
        self.model_btn.setObjectName("primary")
        self.model_btn.clicked.connect(self.build_3d_model)
        self.model_btn.setEnabled(False)
        modeling_layout.addWidget(self.model_btn)

        modeling_group.setLayout(modeling_layout)
        layout.addWidget(modeling_group)

        # 交互与分析
        interact_group = QGroupBox("交互与分析")
        interact_layout = QVBoxLayout()
        interact_layout.setSpacing(10)

        # Z轴拉伸
        interact_layout.addWidget(QLabel("垂直夸张 (Z-Scale):"))
        z_scale_layout = QHBoxLayout()
        self.z_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_scale_slider.setRange(10, 100) # 1.0 - 10.0
        self.z_scale_slider.setValue(10)
        # 优化：使用 sliderReleased 避免滑动时频繁重绘
        self.z_scale_slider.sliderReleased.connect(lambda: self.on_z_scale_changed(self.z_scale_slider.value()))
        # 仅更新标签显示，不触发重绘
        self.z_scale_slider.valueChanged.connect(lambda v: self.z_scale_label.setText(f"{v/10.0:.1f}x"))
        self.z_scale_label = QLabel("1.0x")
        z_scale_layout.addWidget(self.z_scale_slider)
        z_scale_layout.addWidget(self.z_scale_label)
        interact_layout.addLayout(z_scale_layout)

        # 剖面切割
        self.slice_cb = QCheckBox("启用剖面切割")
        self.slice_cb.stateChanged.connect(self.on_slice_toggled)
        interact_layout.addWidget(self.slice_cb)
        
        self.slice_controls = QWidget()
        slice_layout = QVBoxLayout(self.slice_controls)
        slice_layout.setContentsMargins(0,0,0,0)
        
        slice_layout.addWidget(QLabel("切割方向:"))
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems(['X轴', 'Y轴', 'Z轴', '任意'])
        self.slice_axis_combo.currentTextChanged.connect(self.on_slice_axis_changed)
        slice_layout.addWidget(self.slice_axis_combo)
        
        slice_layout.addWidget(QLabel("位置:"))
        self.slice_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_pos_slider.setRange(0, 100)
        self.slice_pos_slider.setValue(50)
        # 优化：切割位置也使用释放触发，防止卡顿
        self.slice_pos_slider.sliderReleased.connect(lambda: self.on_slice_pos_changed(self.slice_pos_slider.value()))
        slice_layout.addWidget(self.slice_pos_slider)
        
        self.interactive_slice_cb = QCheckBox("交互式手柄")
        self.interactive_slice_cb.stateChanged.connect(self.on_interactive_slice_toggled)
        slice_layout.addWidget(self.interactive_slice_cb)
        
        self.slice_controls.setVisible(False)
        interact_layout.addWidget(self.slice_controls)

        # 钻孔拾取
        self.pick_borehole_cb = QCheckBox("启用钻孔点击")
        self.pick_borehole_cb.stateChanged.connect(self.on_pick_mode_toggled)
        interact_layout.addWidget(self.pick_borehole_cb)

        # 测量工具
        self.measure_btn = QPushButton("测量距离")
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.toggle_measure_mode)
        interact_layout.addWidget(self.measure_btn)

        interact_group.setLayout(interact_layout)
        layout.addWidget(interact_group)

        # 渲染控制
        render_group = QGroupBox("渲染控制")
        render_layout = QVBoxLayout()
        render_layout.setSpacing(10)

        render_layout.addWidget(QLabel("显示地层:"))

        # 地层选择工具栏 - 改进
        layer_toolbar = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setMaximumWidth(60)
        self.select_all_btn.setFont(QFont("Microsoft YaHei", 9))
        self.select_all_btn.clicked.connect(self.select_all_layers)

        self.select_none_btn = QPushButton("清空")
        self.select_none_btn.setMaximumWidth(60)
        self.select_none_btn.setFont(QFont("Microsoft YaHei", 9))
        self.select_none_btn.clicked.connect(self.deselect_all_layers)

        self.invert_selection_btn = QPushButton("反选")
        self.invert_selection_btn.setMaximumWidth(60)
        self.invert_selection_btn.setFont(QFont("Microsoft YaHei", 9))
        self.invert_selection_btn.clicked.connect(self.invert_layer_selection)

        layer_toolbar.addWidget(self.select_all_btn)
        layer_toolbar.addWidget(self.select_none_btn)
        layer_toolbar.addWidget(self.invert_selection_btn)
        layer_toolbar.addStretch()
        render_layout.addLayout(layer_toolbar)

        # 搜索框
        self.layer_search = QLineEdit()
        self.layer_search.setPlaceholderText("搜索地层...")
        self.layer_search.textChanged.connect(self.filter_layers)
        self.layer_search.setMaximumHeight(28)
        render_layout.addWidget(self.layer_search)

        # 地层列表 - 改进样式
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(200)
        self.layer_list.setMinimumHeight(150)
        # 使用 NoSelection 模式，完全依赖 CheckBox
        self.layer_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.layer_list.itemChanged.connect(self.on_layer_item_changed)
        
        # 启用右键菜单
        self.layer_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self.show_layer_context_menu)
        
        # 设置样式
        self.layer_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #313244;
            }
        """)
        render_layout.addWidget(self.layer_list)

        # 地层统计信息
        self.layer_stats_label = QLabel("地层: 0/0")
        self.layer_stats_label.setStyleSheet("color: #7f849c; font-size: 11px;")
        render_layout.addWidget(self.layer_stats_label)

        render_layout.addWidget(QLabel("渲染模式:"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(['真实纹理', '增强材质', '基础渲染', '线框模式'])
        self.render_mode_combo.currentTextChanged.connect(self.on_render_mode_changed)
        render_layout.addWidget(self.render_mode_combo)

        render_layout.addWidget(QLabel("透明度:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(90)
        # 优化：透明度调整使用释放触发
        self.opacity_slider.sliderReleased.connect(lambda: self.on_opacity_changed(self.opacity_slider.value()))
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v/100.0:.2f}"))
        self.opacity_label = QLabel("0.90")
        render_layout.addWidget(self.opacity_slider)
        render_layout.addWidget(self.opacity_label)

        self.show_sides_cb = QCheckBox("显示侧面")
        self.show_sides_cb.setChecked(True)
        self.show_sides_cb.stateChanged.connect(self.on_sides_toggled)
        render_layout.addWidget(self.show_sides_cb)

        self.show_edges_cb = QCheckBox("显示网格")
        self.show_edges_cb.setChecked(True)  # 默认显示网格
        self.show_edges_cb.stateChanged.connect(self.on_edges_toggled)
        render_layout.addWidget(self.show_edges_cb)

        self.show_boreholes_cb = QCheckBox("显示钻孔")
        self.show_boreholes_cb.setChecked(False)
        self.show_boreholes_cb.stateChanged.connect(self.on_boreholes_toggled)
        render_layout.addWidget(self.show_boreholes_cb)

        refresh_btn = QPushButton("🔄 刷新渲染")
        refresh_btn.clicked.connect(self.refresh_render)
        render_layout.addWidget(refresh_btn)

        render_group.setLayout(render_layout)
        layout.addWidget(render_group)

        # 高级功能
        advanced_group = QGroupBox("高级功能")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(10)

        # 等值线
        self.contour_cb = QCheckBox("显示等值线")
        self.contour_cb.stateChanged.connect(self.on_contour_toggled)
        advanced_layout.addWidget(self.contour_cb)

        self.contour_params_widget = QWidget()
        contour_layout = QVBoxLayout(self.contour_params_widget)
        contour_layout.setContentsMargins(0, 0, 0, 0)
        
        contour_layout.addWidget(QLabel("类型:"))
        self.contour_type_combo = QComboBox()
        self.contour_type_combo.addItems(['底板高程', '地层厚度'])
        self.contour_type_combo.currentTextChanged.connect(self.on_contour_params_changed)
        contour_layout.addWidget(self.contour_type_combo)

        contour_layout.addWidget(QLabel("间距(m):"))
        self.contour_interval_spin = QDoubleSpinBox()
        self.contour_interval_spin.setRange(1.0, 100.0)
        self.contour_interval_spin.setValue(10.0)
        self.contour_interval_spin.valueChanged.connect(self.on_contour_params_changed)
        contour_layout.addWidget(self.contour_interval_spin)
        
        self.contour_params_widget.setVisible(False)
        advanced_layout.addWidget(self.contour_params_widget)

        # 漫游模式
        self.fly_mode_cb = QCheckBox("虚拟漫游模式 (WASD)")
        self.fly_mode_cb.stateChanged.connect(self.on_fly_mode_toggled)
        advanced_layout.addWidget(self.fly_mode_cb)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # 导出
        export_group = QGroupBox("💾 导出")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(10)

        self.export_png_btn = QPushButton("PNG截图")
        self.export_png_btn.setObjectName("success")
        self.export_png_btn.clicked.connect(lambda: self.export_model('png'))
        self.export_png_btn.setEnabled(False)
        export_layout.addWidget(self.export_png_btn)

        self.export_html_btn = QPushButton("HTML交互")
        self.export_html_btn.setObjectName("success")
        self.export_html_btn.clicked.connect(lambda: self.export_model('html'))
        self.export_html_btn.setEnabled(False)
        export_layout.addWidget(self.export_html_btn)

        self.export_obj_btn = QPushButton("OBJ模型")
        self.export_obj_btn.setObjectName("success")
        self.export_obj_btn.clicked.connect(lambda: self.export_model('obj'))
        self.export_obj_btn.setEnabled(False)
        export_layout.addWidget(self.export_obj_btn)

        self.export_stl_btn = QPushButton("STL模型")
        self.export_stl_btn.setObjectName("success")
        self.export_stl_btn.clicked.connect(lambda: self.export_model('stl'))
        self.export_stl_btn.setEnabled(False)
        export_layout.addWidget(self.export_stl_btn)

        self.export_vtk_btn = QPushButton("VTK模型")
        self.export_vtk_btn.setObjectName("success")
        self.export_vtk_btn.clicked.connect(lambda: self.export_model('vtk'))
        self.export_vtk_btn.setEnabled(False)
        export_layout.addWidget(self.export_vtk_btn)

        self.export_flac3d_btn = QPushButton("FLAC3D网格")
        self.export_flac3d_btn.setObjectName("success")
        self.export_flac3d_btn.clicked.connect(lambda: self.export_model('flac3d'))
        self.export_flac3d_btn.setEnabled(False)
        export_layout.addWidget(self.export_flac3d_btn)

        # FLAC3D 降采样选项
        export_layout.addWidget(QLabel("FLAC3D降采样:"))
        self.flac3d_downsample_spin = QSpinBox()
        self.flac3d_downsample_spin.setRange(1, 10)
        self.flac3d_downsample_spin.setValue(1)
        self.flac3d_downsample_spin.setSuffix("x")
        self.flac3d_downsample_spin.setToolTip("降采样因子：2x减少75%网格，3x减少89%网格\n推荐：大模型使用2-3x，小模型使用1x")
        export_layout.addWidget(self.flac3d_downsample_spin)

        # FLAC3D 格式选择
        export_layout.addWidget(QLabel("FLAC3D格式:"))
        self.flac3d_format_combo = QComboBox()
        self.flac3d_format_combo.addItems(['f3grid (推荐)', 'FPN (中间格式)', '紧凑脚本', '完整脚本'])
        self.flac3d_format_combo.setToolTip(
            "f3grid: 原生网格格式，使用 zone import f3grid 导入\n"
            "FPN: Midas GTS NX中间格式，可用转换工具转换为f3grid\n"
            "紧凑脚本: .f3dat 格式，文件小\n"
            "完整脚本: .f3dat 传统格式，兼容性好"
        )
        export_layout.addWidget(self.flac3d_format_combo)

        # 接触面选项（仅对 f3grid 和 FPN 格式有效）
        self.create_interfaces_checkbox = QCheckBox("创建层间接触面 (Interface)")
        self.create_interfaces_checkbox.setToolTip(
            "启用后，层间节点不共享，并生成接触面定义脚本\n"
            "用于模拟层间滑动、分离等接触行为\n"
            "注意：仅对 f3grid 和 FPN 格式有效"
        )
        self.create_interfaces_checkbox.setChecked(False)
        export_layout.addWidget(self.create_interfaces_checkbox)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()
        
        scroll.setWidget(panel)
        container_layout.addWidget(scroll)

        return container

    def create_render_panel(self) -> QWidget:
        """创建中央3D渲染面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("background-color: #252635; border-bottom: 1px solid #45475a;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        title = QLabel("三维视图")
        title.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        header_layout.addWidget(title)
        
        # --- 新增按钮区域 ---
        header_layout.addStretch()
        
        btn_style = """
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #585b70;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
            QPushButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                padding-right: 2px;
                image: none; /* 隐藏默认箭头，手动绘制或忽略 */
            }
        """
        
        # 复位视角
        reset_btn = QPushButton("复位")
        reset_btn.setToolTip("复位到默认视角")
        reset_btn.setStyleSheet(btn_style)
        reset_btn.clicked.connect(lambda: self.plotter.view_isometric() if self.plotter else None)
        header_layout.addWidget(reset_btn)
        
        # 顶视图
        top_btn = QPushButton("顶视")
        top_btn.setToolTip("切换到顶部视角")
        top_btn.setStyleSheet(btn_style)
        top_btn.clicked.connect(lambda: self.plotter.view_xy() if self.plotter else None)
        header_layout.addWidget(top_btn)
        
        # 截图
        shot_btn = QPushButton("截图")
        shot_btn.setToolTip("保存当前视图截图")
        shot_btn.setStyleSheet(btn_style)
        shot_btn.clicked.connect(lambda: self.export_model('png'))
        header_layout.addWidget(shot_btn)
        
        # 导出菜单
        export_btn = QPushButton("导出 ▼")
        export_btn.setToolTip("导出模型数据")
        export_btn.setStyleSheet(btn_style)
        
        export_menu = QMenu(self)
        export_menu.setStyleSheet("""
            QMenu {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #45475a;
            }
        """)
        
        actions = [
            ("导出 VTK", 'vtk'),
            ("导出 OBJ", 'obj'),
            ("导出 STL", 'stl'),
            ("导出 FLAC3D", 'flac3d'),
            ("导出 HTML", 'html')
        ]
        
        for label, fmt in actions:
            action = QAction(label, self)
            # 使用闭包捕获 fmt
            action.triggered.connect(lambda checked, f=fmt: self.export_model(f))
            export_menu.addAction(action)
            
        export_btn.setMenu(export_menu)
        header_layout.addWidget(export_btn)
        # -------------------

        layout.addWidget(header)

        if PYVISTA_AVAILABLE:
            self.plotter = QtInteractor(panel)
            self.plotter.set_background('#181825') # 深色背景
            layout.addWidget(self.plotter.interactor)
            self.plotter.add_axes()
            
            # 启用鼠标坐标追踪
            self.plotter.track_mouse_position()
            # 使用PyVista的事件系统来跟踪鼠标移动
            self.plotter.iren.add_observer('MouseMoveEvent', self._on_mouse_move_event)
            
            self.log("✓ PyVista GPU渲染器已启用")
        else:
            placeholder = QLabel("PyVista未安装\n请运行: pip install pyvistaqt")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("font-size: 16px; color: #f38ba8;")
            layout.addWidget(placeholder)
            self.plotter = None

        return panel

    def create_info_panel(self) -> QWidget:
        """创建右侧信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("统计与日志")
        title.setObjectName("header")
        layout.addWidget(title)

        # 使用 QTextEdit 替换 QLabel 以支持滚动，防止内容过多撑爆窗口
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText("等待加载数据...")
        self.stats_text.setStyleSheet("""
            QTextEdit {
                color: #a6adc8; 
                background-color: #313244; 
                padding: 8px; 
                border-radius: 6px; 
                border: 1px solid #45475a;
                font-family: "Consolas", "Microsoft YaHei";
                font-size: 13px;
            }
        """)
        layout.addWidget(self.stats_text, 1) # 权重1

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("控制台输出:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.log_text.customContextMenuRequested.connect(self.show_log_context_menu)
        self.log_text.setStyleSheet("""
            QTextEdit {
                color: #cdd6f4;
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 6px;
                font-family: "Consolas", monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.log_text, 2) # 权重2，给日志更多空间

        return panel

    def show_log_context_menu(self, position):
        """显示日志右键菜单"""
        menu = QMenu()
        
        action_copy = QAction("复制", self)
        action_copy.triggered.connect(self.log_text.copy)
        menu.addAction(action_copy)
        
        action_clear = QAction("清空日志", self)
        action_clear.triggered.connect(self.log_text.clear)
        menu.addAction(action_clear)
        
        menu.exec(self.log_text.mapToGlobal(position))

    def on_method_changed(self):
        """预测方法切换"""
        use_trad = self.traditional_radio.isChecked()
        use_gnn = self.gnn_radio.isChecked()

        if use_trad and use_gnn:
            sender = self.sender()
            if sender == self.traditional_radio:
                self.gnn_radio.setChecked(False)
            else:
                self.traditional_radio.setChecked(False)

        self.traditional_params.setVisible(self.traditional_radio.isChecked())
        self.gnn_params.setVisible(self.gnn_radio.isChecked())

    def check_gpu(self):
        """检查GPU可用性"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.log(f"✓ GPU检测: {gpu_name} ({gpu_memory:.1f} GB)")
            self.statusBar().showMessage(f"就绪 | GPU: {gpu_name}")
        else:
            self.log("⚠️ 未检测到CUDA GPU，将使用CPU")
            self.statusBar().showMessage("就绪 | GPU: 不可用 (CPU模式)")

    def append_log(self, message: str):
        """添加日志"""
        if self.log_text is not None:
            self.log_text.append(message)
            self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        else:
            print(message)

    def log(self, message: str):
        """Legacy log method wrapper"""
        logging.info(message)

    def set_busy_state(self, is_busy: bool):
        """设置忙碌状态"""
        if is_busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.control_panel.setEnabled(False)
            self.menuBar().setEnabled(False)
        else:
            QApplication.restoreOverrideCursor()
            self.control_panel.setEnabled(True)
            self.menuBar().setEnabled(True)

    def load_data(self):
        """加载数据"""
        self.log("\n" + "="*50)
        self.log("开始加载数据...")

        self.set_busy_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.data_loader = DataLoaderThread(
            data_dir=str(self.data_dir),
            merge_coal=self.merge_coal_cb.isChecked(),
            k_neighbors=self.k_neighbors_spin.value(),
            layer_method=self.layer_method_combo.currentText(),
            min_occurrence_rate=self.min_occurrence_spin.value()
        )

        self.data_loader.progress.connect(self.log)
        self.data_loader.finished.connect(self.on_data_loaded)
        self.data_loader.error.connect(self.on_error)

        self.data_loader.start()

    def on_data_loaded(self, result: dict):
        """数据加载完成"""
        self.data_result = result
        self.set_busy_state(False)
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 自动计算基准面: 优先使用原始数据中的z/高程字段
        raw_df = result.get('raw_df')
        auto_base = None
        if raw_df is not None:
            for col in ['z', 'elevation', 'top_depth', 'bottom_depth']:
                if col in raw_df.columns:
                    try:
                        vals = raw_df[col].astype(float)
                        if len(vals) > 0:
                            auto_base = float(vals.min())
                            break
                    except Exception:
                        pass
        if auto_base is not None and hasattr(self, 'base_level_spin'):
            self.base_level_spin.setValue(auto_base)
            self.log(f"✓ 自动基准面: {auto_base:.2f} (来自数据最小值)")
        else:
            self.log("⚠️ 未找到z/高程字段，基准面保持默认0")

        stats = f"""
📊 数据统计:
- 钻孔数: {len(result['borehole_ids'])}
- 地层数: {result['num_layers']}
- 特征维度: {result['num_features']}

地层序列 (底→顶):
"""
        for i, layer in enumerate(result['layer_order']):
            stats += f"{i+1}. {layer} ({result['exist_rate'][i]*100:.0f}%)\n"

        self.stats_text.setText(stats)
        self.log("✓ 数据加载完成，可以开始训练")

    def train_model(self):
        """训练模型"""
        if self.data_result is None:
            QMessageBox.warning(self, "警告", "请先加载数据!")
            return
            
        # Input validation
        if len(self.data_result['borehole_ids']) < 3:
            QMessageBox.warning(self, "警告", "钻孔数量过少 (<3)，无法进行有效训练或插值。")
            return

        self.log("\n" + "="*50)

        # 智能选择：小样本自动切换传统方法
        n_bh = len(self.data_result.get('borehole_ids', [])) if self.data_result else 0
        recommended = None
        if n_bh < 5:
            recommended = 'constant'
        elif n_bh < 15:
            recommended = 'idw'
        # elif n_bh < 50:
        #     recommended = 'kriging'

        use_traditional = self.traditional_radio.isChecked()

        if recommended is not None:
            if not use_traditional:
                self.traditional_radio.setChecked(True)
                self.gnn_radio.setChecked(False)
                use_traditional = True
            self.log(f"⚠️ 钻孔样本较少({n_bh})，自动使用传统方法: {recommended}")
            if hasattr(self, 'interp_method_combo'):
                self.interp_method_combo.setCurrentText('kriging' if recommended == 'kriging' else 'idw')

        if use_traditional:
            self.train_traditional()
        else:
            self.train_gnn()

    def train_traditional(self):
        """传统方法拟合"""
        self.log("使用传统地质统计学方法...")
        self.use_traditional = True

        # 创建进度对话框
        self.progress_dialog = ModernProgressDialog(
            self,
            "模型训练",
            "正在初始化传统插值模型..."
        )
        self.progress_dialog.set_indeterminate(False)
        self.progress_dialog.set_progress(0)

        self.set_busy_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.trainer = TraditionalPredictorThread(
            data_result=self.data_result,
            interp_method=self.interp_method_combo.currentText()
        )

        self.trainer.progress.connect(self.log)
        self.trainer.progress_percent.connect(self._on_training_progress)
        self.trainer.finished.connect(self.on_traditional_trained)
        self.trainer.error.connect(self.on_error)

        # 显示进度对话框
        self.progress_dialog.show()
        self.trainer.start()

    def on_traditional_trained(self, predictor, metrics):
        """传统方法拟合完成"""
        self.predictor = predictor
        self.model = None
        self.use_traditional = True

        self.set_busy_state(False)
        self.model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 关闭进度对话框
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_progress(100)
            self.progress_dialog.set_message("✓ 训练完成!")
            self.progress_dialog.auto_close_on_complete()

        stats = f"""
✓ 传统方法拟合完成

评估指标:
- MAE: {metrics.get('mae', 0):.3f} m
- RMSE: {metrics.get('rmse', 0):.3f} m
- R²: {metrics.get('r2', 0):.3f}
- MAPE: {metrics.get('mape', 0):.1f}%
"""
        self.log(stats)

    def train_gnn(self):
        """GNN训练"""
        self.log("使用GNN深度学习方法...")
        self.use_traditional = False

        n_samples = self.data_result['data'].x.shape[0]
        n_layers = self.data_result['num_layers']
        n_features = self.data_result['num_features']

        if self.auto_config_cb.isChecked():
            opt_config = get_optimized_config_for_small_dataset(
                n_samples=n_samples,
                n_layers=n_layers,
                n_features=n_features
            )
            config = {
                'num_features': n_features,
                'num_layers': n_layers,
                'hidden_dim': opt_config['model']['hidden_channels'],
                'gnn_layers': opt_config['model']['num_layers'],
                'dropout': opt_config['model']['dropout'],
                'conv_type': 'gatv2',
                'learning_rate': opt_config['trainer']['learning_rate'],
                'epochs': opt_config['training']['epochs'],
                'patience': opt_config['training']['patience'],
                'use_augmentation': opt_config['trainer']['use_augmentation'],
                'warmup_epochs': opt_config['training']['warmup_epochs'],
                'heads': opt_config['model'].get('heads', 4)
            }
            self.log(f"自动配置: hidden={config['hidden_dim']}, layers={config['gnn_layers']}")
        else:
            config = {
                'num_features': n_features,
                'num_layers': n_layers,
                'hidden_dim': 128,
                'gnn_layers': 3,
                'dropout': 0.2,
                'conv_type': 'gatv2',
                'learning_rate': 0.001,
                'epochs': self.epochs_spin.value(),
                'patience': 30,
                'heads': 4
            }

        # 创建进度对话框
        self.progress_dialog = ModernProgressDialog(
            self,
            "GNN模型训练",
            "正在初始化神经网络模型..."
        )
        self.progress_dialog.set_indeterminate(False)
        self.progress_dialog.set_progress(0)
        self.progress_dialog.set_detail(f"训练轮数: 0/{config['epochs']}")

        self.set_busy_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, config['epochs'])

        self.trainer = TrainingThread(
            data_result=self.data_result,
            config=config
        )

        self.trainer.progress.connect(self.log)
        self.trainer.progress_percent.connect(self._on_training_progress)
        self.trainer.finished.connect(self.on_gnn_trained)
        self.trainer.error.connect(self.on_error)

        # 显示进度对话框
        self.progress_dialog.show()
        self.trainer.start()

    def on_gnn_trained(self, model, history):
        """GNN训练完成"""
        self.model = model
        self.predictor = model
        self.use_traditional = False

        self.set_busy_state(False)
        self.model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 关闭进度对话框
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_progress(100)
            self.progress_dialog.set_message("✓ 训练完成!")
            self.progress_dialog.auto_close_on_complete()

        if 'test_metrics' in history:
            metrics = history['test_metrics']
            stats = f"""
✓ GNN训练完成

测试集评估:
- MAE: {metrics['mae']:.3f} m
- RMSE: {metrics['rmse']:.3f} m
- R²: {metrics['r2']:.3f}

训练轮数: {len(history['train_loss'])}
"""
            self.log(stats)

    def build_3d_model(self):
        """构建三维模型"""
        if self.predictor is None:
            QMessageBox.warning(self, "警告", "请先训练模型!")
            return
            
        # 清除旧的网格缓存，防止数据不一致
        self.mesh_cache = {}
        self.cached_meshes = {}
        
        # Input validation
        resolution = self.resolution_spin.value()
        if resolution > 500:
            reply = QMessageBox.question(
                self, "高分辨率警告", 
                f"当前分辨率 ({resolution}) 较高，可能会导致内存溢出或计算缓慢。\n是否继续?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.log("\n" + "="*50)
        self.log("开始构建三维模型...")

        # 创建进度对话框
        self.progress_dialog = ModernProgressDialog(
            self,
            "三维建模",
            "正在初始化建模参数..."
        )
        self.progress_dialog.set_indeterminate(False)
        self.progress_dialog.set_progress(0)
        self.progress_dialog.set_detail(f"分辨率: {resolution} × {resolution}")

        self.set_busy_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.modeler = ModelingThread(
            data_result=self.data_result,
            predictor=self.predictor,
            resolution=self.resolution_spin.value(),
            base_level=self.base_level_spin.value(),
            gap_value=0.0,
            use_traditional=self.use_traditional
        )

        self.modeler.progress.connect(self.log)
        self.modeler.progress_percent.connect(self._on_modeling_progress)
        self.modeler.finished.connect(self.on_model_built)
        self.modeler.error.connect(self.on_error)

        # 显示进度对话框
        self.progress_dialog.show()
        self.modeler.start()

    def on_resolution_changed(self, value):
        """分辨率改变 - 延迟自动重建"""
        if self.predictor is not None:
            self.resolution_timer.start()

    def on_base_level_changed(self, value):
        """基准面改变 - 实时平移"""
        if not self.plotter or not self.block_models:
            return
            
        delta = value - self.last_base_level
        self.last_base_level = value
        
        # 平移所有Actor
        for actor in self.plotter.actors.values():
            if hasattr(actor, 'SetPosition'):
                pos = actor.GetPosition()
                actor.SetPosition(pos[0], pos[1], pos[2] + delta)
        
        self.plotter.render()

    def on_z_scale_changed(self, value):
        """Z轴缩放改变"""
        scale = value / 10.0
        self.z_scale_label.setText(f"{scale:.1f}x")
        if self.plotter:
            self.plotter.set_scale(zscale=scale)

    def on_slice_toggled(self, state):
        """剖面切割开关"""
        is_checked = (state == Qt.CheckState.Checked.value)
        self.slice_controls.setVisible(is_checked)
        self.request_render()

    def on_slice_axis_changed(self, text):
        """切割轴改变"""
        if self.slice_cb.isChecked():
            self.request_render()

    def on_slice_pos_changed(self, value):
        """切割位置改变"""
        if not self.slice_cb.isChecked() or not hasattr(self, 'active_plane_widget'):
            return
            
        # 更新切割平面位置
        axis = self.slice_axis_combo.currentText()
        if axis == '任意':
            return
            
        # 获取模型边界
        bounds = self.plotter.bounds
        # bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        
        pos_ratio = value / 100.0
        
        origin = list(self.plotter.center)
        normal = (1, 0, 0)
        
        if axis == 'X轴':
            origin[0] = bounds[0] + (bounds[1] - bounds[0]) * pos_ratio
            normal = (1, 0, 0)
        elif axis == 'Y轴':
            origin[1] = bounds[2] + (bounds[3] - bounds[2]) * pos_ratio
            normal = (0, 1, 0)
        elif axis == 'Z轴':
            origin[2] = bounds[4] + (bounds[5] - bounds[4]) * pos_ratio
            normal = (0, 0, 1)
            
        # 更新平面部件
        if self.active_plane_widget:
            self.active_plane_widget.SetOrigin(origin)
            self.active_plane_widget.SetNormal(normal)
            self.active_plane_widget.UpdatePlacement()
            self.plotter.render()

    def on_interactive_slice_toggled(self, state):
        """交互式切割切换"""
        is_checked = (state == Qt.CheckState.Checked.value)
        self.slice_axis_combo.setEnabled(not is_checked)
        self.slice_pos_slider.setEnabled(not is_checked)
        
        if is_checked:
            # 切换到任意方向以启用交互式手柄
            self.slice_axis_combo.setCurrentText('任意')
        else:
            # 恢复默认
            if self.slice_axis_combo.currentText() == '任意':
                self.slice_axis_combo.setCurrentText('X轴')
            
        self.request_render()

    def on_pick_mode_toggled(self, state):
        """钻孔拾取开关"""
        if state == Qt.CheckState.Checked.value:
            self.plotter.enable_point_picking(callback=self.on_borehole_picked, show_message=False, show_point=False)
            self.log("已启用钻孔拾取: 请点击红色钻孔标记")
        else:
            self.plotter.disable_picking()
            self.log("已禁用钻孔拾取")

    def _on_mouse_move_event(self, _obj, _event):
        """VTK鼠标移动事件回调，获取3D坐标并调用on_mouse_move"""
        try:
            # 获取鼠标在屏幕上的位置
            x, y = self.plotter.iren.GetEventPosition()
            # 使用pick方法获取3D世界坐标
            picker = self.plotter.iren.GetPicker()
            if picker and picker.Pick(x, y, 0, self.plotter.renderer):
                point = picker.GetPickPosition()
                self.on_mouse_move(point)
        except Exception:
            pass  # 静默处理拾取错误

    def on_mouse_move(self, point):
        """鼠标移动回调，更新状态栏坐标和地层信息"""
        if not point:
            return

        info = f"X: {point[0]:.2f}, Y: {point[1]:.2f}, Z: {point[2]:.2f}"
        
        # 获取当前鼠标下的Actor
        # track_mouse_position 会更新 picked_actor
        actor = self.plotter.picked_actor
        
        layer_name = None
        if actor:
            # 反查Actor名称
            for name, a in self.plotter.actors.items():
                if a == actor:
                    layer_name = name
                    break
        
        if layer_name:
            # 处理名称 (去除 _sides 后缀)
            display_name = layer_name.replace("_sides", "")
            
            # 确认是地层 (排除钻孔、辅助线等)
            is_layer = False
            if self.block_models:
                for bm in self.block_models:
                    if bm.name == display_name:
                        is_layer = True
                        break
            
            if is_layer:
                info += f" | 📍 地层: {display_name}"
                
                # 在左上角显示悬浮标签
                text = f"当前地层: {display_name}"
                if 'hover_layer_label' in self.plotter.actors:
                    # 仅当文本变化时更新，避免闪烁
                    current_actor = self.plotter.actors['hover_layer_label']
                    # PyVista的Actor包装器可能没有GetInput，尝试直接访问mapper或input
                    # 这里简单处理：总是更新，但SetInput开销很小
                    try:
                        current_actor.SetInput(text)
                        current_actor.SetVisibility(True)
                    except:
                        pass # 忽略可能的属性错误
                else:
                    # 创建新标签
                    self.plotter.add_text(
                        text,
                        position=(20, 20),
                        font_size=16,
                        color='#cdd6f4', # 与主题一致的淡紫色
                        name='hover_layer_label',
                        shadow=True
                    )
            else:
                # 隐藏标签
                if 'hover_layer_label' in self.plotter.actors:
                    self.plotter.actors['hover_layer_label'].SetVisibility(False)
        else:
            # 隐藏标签
            if 'hover_layer_label' in self.plotter.actors:
                self.plotter.actors['hover_layer_label'].SetVisibility(False)
            
        self.statusBar().showMessage(info)

    def on_borehole_picked(self, point, actor):
        """钻孔被点击"""
        if not self.data_result or 'borehole_coords' not in self.data_result:
            return
            
        # 查找最近的钻孔
        coords = self.data_result['borehole_coords']
        ids = self.data_result['borehole_ids']
        
        # 只比较X,Y距离，忽略Z
        dists = np.sqrt((coords[:, 0] - point[0])**2 + (coords[:, 1] - point[1])**2)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist > 50: # 阈值，避免误触
            return
            
        bh_id = ids[min_idx]
        self.log(f"选中钻孔: {bh_id}")
        
        # 显示详情
        if 'raw_df' in self.data_result:
            df = self.data_result['raw_df']
            bh_data = df[df['borehole_id'] == bh_id].sort_values('layer_order')
            
            dialog = BoreholeInfoDialog(bh_id, bh_data, self)
            dialog.show()

    def toggle_measure_mode(self):
        """切换测量模式"""
        if self.measure_btn.isChecked():
            self.pick_borehole_cb.setChecked(False)
            self.measure_points = []
            self.plotter.enable_point_picking(callback=self.on_measure_picked, show_message=True, font_size=10, color='pink', point_size=10, use_picker=True)
            self.log("📏 测量模式: 请点击两个点进行测量")
        else:
            self.plotter.disable_picking()
            self.plotter.clear_measure_widgets() # If available
            # Remove markers
            self.plotter.remove_actor('measure_p1')
            self.plotter.remove_actor('measure_p2')
            self.plotter.remove_actor('measure_line')
            self.log("已退出测量模式")

    def on_measure_picked(self, point, actor):
        """测量点拾取回调"""
        # 如果已经有两个点，重置开始新的测量
        if len(self.measure_points) >= 2:
            self.measure_points = []
            self.plotter.remove_actor('measure_p1')
            self.plotter.remove_actor('measure_p2')
            self.plotter.remove_actor('measure_line')
            self.plotter.remove_actor('measure_label')

        self.measure_points.append(point)
        
        if len(self.measure_points) == 1:
            self.plotter.add_mesh(
                pv.PolyData(point), color='yellow', point_size=15, 
                render_points_as_spheres=True, name='measure_p1'
            )
            self.log(f"起点: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
            
        elif len(self.measure_points) == 2:
            p1 = self.measure_points[0]
            p2 = point
            
            self.plotter.add_mesh(
                pv.PolyData(p2), color='yellow', point_size=15, 
                render_points_as_spheres=True, name='measure_p2'
            )
            
            # Draw line
            line = pv.Line(p1, p2)
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            dz = abs(p1[2] - p2[2])
            dxy = np.sqrt(dist**2 - dz**2)
            
            self.plotter.add_mesh(
                line, color='yellow', line_width=4, name='measure_line'
            )
            
            # Add label at midpoint
            mid_point = (np.array(p1) + np.array(p2)) / 2
            label = f"距离: {dist:.2f}m\n水平: {dxy:.2f}m\n垂直: {dz:.2f}m"
            
            self.plotter.add_point_labels(
                [mid_point], [label],
                point_size=0, font_size=16, text_color='yellow',
                show_points=False, name='measure_label',
                always_visible=True, shape_opacity=0.5
            )
            
            self.log(f"终点: ({p2[0]:.1f}, {p2[1]:.1f}, {p2[2]:.1f})")
            self.log(f"📏 测量结果: 距离={dist:.2f}m (水平={dxy:.2f}m, 垂直={dz:.2f}m)")
            self.log(f"📏 距离: {dist:.2f} m")
            
            # Reset for next measurement
            self.measure_points = []

    def on_contour_toggled(self, state):
        """等值线开关 - 实时"""
        is_checked = (state == Qt.CheckState.Checked.value)
        self.contour_params_widget.setVisible(is_checked)
        self.update_contours()

    def on_contour_params_changed(self):
        """等值线参数改变 - 实时"""
        if self.contour_cb.isChecked():
            self.update_contours()

    def on_fly_mode_toggled(self, state):
        """漫游模式开关"""
        if not self.plotter:
            return
            
        if state == Qt.CheckState.Checked.value:
            self.plotter.enable_terrain_style(mouse_wheel_zooms=True)
            self.log("已启用漫游模式: 左键旋转，中键平移，右键缩放/前进")
        else:
            self.plotter.enable_trackball_style()
            self.log("已恢复标准视图模式")

    def on_model_built(self, block_models, XI, YI):
        """三维模型构建完成"""
        self.block_models = block_models
        self.XI = XI
        self.YI = YI
        self.last_base_level = self.base_level_spin.value()

        # 清空渲染缓存
        self.cached_meshes = {}
        self.cached_textures = {}
        self.cached_sides_state = None
        self.mesh_cache = {}
        self.merged_mesh_cache = None  # 清空剖面合并网格缓存

        self.set_busy_state(False)
        self.progress_bar.setVisible(False)

        # 关闭进度对话框
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_progress(100)
            self.progress_dialog.set_message("✓ 建模完成!")
            self.progress_dialog.auto_close_on_complete()

        stats = "✓ 三维模型构建完成\n\n各层统计:\n"
        for bm in block_models:
            stats += f"- {bm.name}: 平均厚度 {bm.avg_thickness:.2f}m\n"

        self.log(stats)

        # 填充地层列表，使用复选框
        self.layer_list.clear()
        for bm in block_models:
            item = QListWidgetItem(bm.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.layer_list.addItem(item)

        # 更新统计
        self.update_layer_stats()

        if PYVISTA_AVAILABLE and self.plotter is not None:
            self.render_3d_model()

        self.export_png_btn.setEnabled(True)
        self.export_html_btn.setEnabled(True)
        self.export_obj_btn.setEnabled(True)
        self.export_stl_btn.setEnabled(True)
        self.export_vtk_btn.setEnabled(True)
        self.export_flac3d_btn.setEnabled(True)

    def update_contours(self):
        """更新等值线显示"""
        if not self.plotter or not self.block_models:
            return

        # 先移除旧的等值线
        for bm in self.block_models:
            self.plotter.remove_actor(f"{bm.name}_contour")

        if not self.contour_cb.isChecked():
            return

        contour_type = self.contour_type_combo.currentText()
        interval = self.contour_interval_spin.value()
        
        # 获取可见层
        visible_layers = set()
        if hasattr(self, 'layer_list'):
            for i in range(self.layer_list.count()):
                item = self.layer_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    visible_layers.add(item.text())
        
        for bm in self.block_models:
            if bm.name not in visible_layers:
                continue
            
            try:
                # 构建网格用于计算等值线
                # 使用顶板作为显示位置，这样等值线浮在层面上方
                grid = pv.StructuredGrid(self.XI, self.YI, bm.top_surface)
                
                scalars_name = ""
                if contour_type == '底板高程':
                    scalars_name = "Elevation"
                    grid.point_data[scalars_name] = bm.bottom_surface.flatten()
                else: # 地层厚度
                    scalars_name = "Thickness"
                    thickness = bm.top_surface - bm.bottom_surface
                    grid.point_data[scalars_name] = thickness.flatten()
                
                # 计算等值线数值范围
                data_min = grid.point_data[scalars_name].min()
                data_max = grid.point_data[scalars_name].max()
                
                if data_max > data_min:
                    # 生成等值线值
                    start_val = np.floor(data_min / interval) * interval
                    levels = np.arange(start_val, data_max, interval)
                    levels = levels[levels >= data_min]
                    
                    if len(levels) > 0:
                        contours = grid.contour(isosurfaces=levels, scalars=scalars_name)
                        
                        line_color = 'white' if contour_type == '底板高程' else 'yellow'
                        
                        self.plotter.add_mesh(
                            contours, 
                            color=line_color, 
                            line_width=3, 
                            name=f"{bm.name}_contour",
                            render_lines_as_tubes=True
                        )
            except Exception as e:
                print(f"等值线生成失败 ({bm.name}): {e}")

    def find_texture_file(self, layer_name: str) -> Optional[Path]:
        """在 textures 目录中按名称模糊匹配贴图文件"""
        if not PYVISTA_AVAILABLE:
            return None
        if not self.texture_dir.exists():
            return None

        name_lower = layer_name.lower()
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        candidates = []
        for path in self.texture_dir.rglob("*"):
            if path.suffix.lower() not in exts:
                continue
            if name_lower in path.stem.lower():
                candidates.append(path)
        if not candidates:
            return None

        # 取最短匹配名，尽量选最贴合的文件
        candidates.sort(key=lambda p: len(p.stem))
        return candidates[0]

    def get_layer_texture(self, layer_name: str, base_color):
        """获取地层贴图，优先本地真实贴图，其次程序纹理"""
        if layer_name in self.cached_textures:
            return self.cached_textures[layer_name]

        texture = None
        tex_path = self.find_texture_file(layer_name)
        if tex_path:
            try:
                texture = pv.read_texture(str(tex_path))
            except Exception as e:
                self.log(f"警告: 读取贴图失败 {tex_path.name}: {e}")

        if texture is None:
            try:
                texture_array = TextureGenerator.generate_rock_texture(
                    layer_name, size=(512, 512), base_color=base_color
                )
                texture = pv.numpy_to_texture(texture_array)
            except Exception as e:
                self.log(f"警告: 程序纹理生成失败: {e}")
                texture = None

        if texture is not None:
            self.cached_textures[layer_name] = texture
        return texture

    def add_legend_safe(self, legend_entries):
        """统一处理图例显示，保证深色背景下文字可见（支持中文）"""
        try:
            self.plotter.remove_legend()
        except Exception:
            pass

        if not legend_entries:
            return

        cleaned = []
        for idx, (name, color) in enumerate(legend_entries):
            # 给标签编号
            label = f"{idx+1}. {name}"
            # 颜色归一化到0-1，并确保是tuple
            if isinstance(color, (tuple, list)) and len(color) == 3:
                r, g, b = color
                if max(r, g, b) > 1.0:
                    color = (r/255.0, g/255.0, b/255.0)
                color = (float(color[0]), float(color[1]), float(color[2]))
            cleaned.append((label, color))

        # 过多条目时只显示前20条，避免拥挤
        max_entries = 20
        if len(cleaned) > max_entries:
            cleaned = cleaned[:max_entries]
            cleaned.append(("... 更多", (0.7, 0.7, 0.7)))

        try:
            # 查找系统中文字体路径
            chinese_font_path = None
            possible_fonts = [
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/simsun.ttc",    # 宋体
                "C:/Windows/Fonts/simkai.ttf",    # 楷体
            ]
            for font_path in possible_fonts:
                if os.path.exists(font_path):
                    chinese_font_path = font_path
                    break
            
            # 使用更美观的图例设置
            legend_actor = self.plotter.add_legend(
                cleaned,
                bcolor='#252635', # 深色背景，与主题一致
                border=True,
                loc='upper left',  # 改到左上角，不遮挡主视图
                size=(0.15, 0.35), # 稍微缩小
                background_opacity=0.85
            )
            
            # 设置字体和颜色
            if legend_actor and hasattr(legend_actor, "GetEntryTextProperty"):
                prop = legend_actor.GetEntryTextProperty()
                prop.SetColor(1.0, 1.0, 1.0) # 纯白
                prop.SetFontSize(14)         # 字体大小
                prop.SetBold(False)          # 取消加粗
                prop.SetShadow(False)
                
                # 设置中文字体（关键修复）
                if chinese_font_path:
                    prop.SetFontFile(chinese_font_path)
                    prop.SetFontFamily(4)  # VTK_FONT_FILE = 4，使用自定义字体文件
                else:
                    # 回退到Arial
                    prop.SetFontFamilyToArial()
                
            # 设置边框颜色
            if legend_actor and hasattr(legend_actor, "GetBorderProperty"):
                legend_actor.GetBorderProperty().SetColor(0.4, 0.4, 0.5)
                
        except Exception as e:
            self.log(f"图例显示失败: {e}")

    def render_3d_model(self):
        """渲染3D模型到PyVista窗口 - 性能优化版"""
        if self.is_rendering:
            return
        self.is_rendering = True
        
        # 暂停渲染以提高性能
        self.plotter.render_window.SetOffScreenRendering(1)

        try:
            # 检查是否需要完全重建场景
            # 如果只是切换可见性，不需要重建
            # 这里我们简化逻辑：如果block_models变了或者渲染模式变了，就重建
            
            # 保存当前相机视角
            camera_pos = self.plotter.camera_position if self.plotter.camera_set else None
            
            # 保存切割平面状态
            current_plane_origin = None
            current_plane_normal = None
            if hasattr(self, 'active_plane_widget') and self.active_plane_widget:
                try:
                    current_plane_origin = self.active_plane_widget.GetOrigin()
                    current_plane_normal = self.active_plane_widget.GetNormal()
                except:
                    pass
            self.active_plane_widget = None

            self.plotter.clear()
            self.plotter.set_background('#181825')
            
            # 启用高级渲染特性
            self.plotter.enable_anti_aliasing()
            # 深度剥离比较耗性能，仅在透明度较低时启用
            opacity = self.opacity_slider.value() / 100.0 if hasattr(self, 'opacity_slider') else 0.9
            if opacity < 0.99:
                self.plotter.enable_depth_peeling()
            else:
                self.plotter.disable_depth_peeling()

            show_sides = self.show_sides_cb.isChecked() if hasattr(self, 'show_sides_cb') else True
            show_edges = self.show_edges_cb.isChecked() if hasattr(self, 'show_edges_cb') else False
            render_mode = self.render_mode_combo.currentText() if hasattr(self, 'render_mode_combo') else '基础渲染'
            enable_slicing = self.slice_cb.isChecked() if hasattr(self, 'slice_cb') else False

            renderer = GeologicalModelRenderer(use_pbr=(render_mode=='增强材质'))

            # 灯光设置
            if render_mode in ['增强材质', '真实纹理']:
                self.plotter.add_light(pv.Light(position=(0, 0, 1000), intensity=0.8))
                self.plotter.add_light(pv.Light(position=(1000, 1000, 1000), intensity=0.5))

            # 缓存网格生成
            cache_key = show_sides
            if cache_key not in self.mesh_cache:
                self.log("正在生成网格几何体...")
                meshes_for_state = {}
                for i, bm in enumerate(self.block_models):
                    color = RockMaterial.get_color(bm.name, i)
                    main_mesh, side_mesh = renderer.create_layer_mesh(
                        self.XI, self.YI,
                        bm.top_surface, bm.bottom_surface,
                        bm.name,
                        color=color,
                        add_sides=show_sides,
                        return_parts=True
                    )
                    meshes_for_state[bm.name] = (main_mesh, side_mesh, color)
                self.mesh_cache[cache_key] = meshes_for_state

            self.cached_meshes = self.mesh_cache[cache_key]
            self.cached_sides_state = show_sides

            legend_entries = []

            # 剖面切割模式 (使用缓存优化)
            if enable_slicing:
                # 获取当前可见层的哈希，用于缓存判断
                visible_layers = []
                for bm in self.block_models:
                    if bm.name not in self.cached_meshes:
                        continue
                    is_visible = True
                    if hasattr(self, 'layer_list'):
                        items = self.layer_list.findItems(bm.name, Qt.MatchFlag.MatchExactly)
                        if items:
                            is_visible = (items[0].checkState() == Qt.CheckState.Checked)
                    if is_visible:
                        visible_layers.append(bm.name)
                
                visible_key = tuple(visible_layers)
                
                # 检查是否可以使用缓存的合并网格
                if (self.merged_mesh_cache is not None and 
                    hasattr(self, '_merged_mesh_visible_key') and 
                    self._merged_mesh_visible_key == visible_key and
                    hasattr(self, '_merged_mesh_sides_key') and
                    self._merged_mesh_sides_key == show_sides):
                    merged_mesh = self.merged_mesh_cache
                    self.log("使用缓存的合并网格...")
                else:
                    # 需要重新合并网格
                    self.log("正在合并网格用于剖面切割...")
                    meshes_to_merge = []
                    for bm in self.block_models:
                        if bm.name not in visible_layers:
                            continue

                        main_mesh, side_mesh, color = self.cached_meshes[bm.name]
                        legend_entries.append((bm.name, color))

                        # 复制并添加颜色标量
                        mesh_copy = main_mesh.copy()
                        if side_mesh and show_sides:
                            mesh_copy = mesh_copy.merge(side_mesh, merge_points=False)

                        rgb_color = (np.array(color) * 255).astype(np.uint8)
                        mesh_copy.point_data['RGB'] = np.tile(rgb_color, (mesh_copy.n_points, 1))
                        meshes_to_merge.append(mesh_copy)
                    
                    if meshes_to_merge:
                        merged_mesh = meshes_to_merge[0].merge(meshes_to_merge[1:], merge_points=False)
                        # 缓存合并网格
                        self.merged_mesh_cache = merged_mesh
                        self._merged_mesh_visible_key = visible_key
                        self._merged_mesh_sides_key = show_sides
                    else:
                        merged_mesh = None
                
                # 填充图例
                if not legend_entries:
                    for bm_name in visible_layers:
                        if bm_name in self.cached_meshes:
                            _, _, color = self.cached_meshes[bm_name]
                            legend_entries.append((bm_name, color))
                
                if merged_mesh is not None:
                    # 切割参数
                    axis = self.slice_axis_combo.currentText()
                    normal = 'x'
                    origin = None
                    if axis == 'Y轴': normal = 'y'
                    elif axis == 'Z轴': normal = 'z'
                    
                    if axis == '任意' and current_plane_normal is not None:
                        normal = current_plane_normal
                        origin = current_plane_origin
                    
                    interaction = self.interactive_slice_cb.isChecked() if hasattr(self, 'interactive_slice_cb') else False
                    interaction_event = "always" if interaction else "end"

                    actor = self.plotter.add_mesh_clip_plane(
                        merged_mesh,
                        normal=normal,
                        origin=origin,
                        scalars='RGB',
                        rgb=True,
                        opacity=opacity,
                        show_edges=show_edges,
                        interaction_event=interaction_event
                    )
                    
                    if hasattr(self.plotter, 'plane_widgets') and self.plotter.plane_widgets:
                        self.active_plane_widget = self.plotter.plane_widgets[-1]
                    
                    if axis != '任意' and not interaction:
                        self.on_slice_pos_changed(self.slice_pos_slider.value())

            else:
                # 标准模式：直接添加Actor，不合并网格 (更快)
                for bm in self.block_models:
                    if bm.name not in self.cached_meshes:
                        continue

                    main_mesh, side_mesh, color = self.cached_meshes[bm.name]
                    
                    # 检查可见性
                    is_visible = True
                    if hasattr(self, 'layer_list'):
                        items = self.layer_list.findItems(bm.name, Qt.MatchFlag.MatchExactly)
                        if items:
                            is_visible = (items[0].checkState() == Qt.CheckState.Checked)
                    
                    if is_visible:
                        legend_entries.append((bm.name, color))
                    
                    # 根据渲染模式添加Actor
                    if render_mode == '线框模式':
                        full_mesh = main_mesh
                        if side_mesh:
                            full_mesh = full_mesh.merge(side_mesh, merge_points=False)
                        actor = self.plotter.add_mesh(
                            full_mesh, color=color, style='wireframe',
                            line_width=2, opacity=opacity * 0.5, name=bm.name
                        )
                    elif render_mode == '真实纹理':
                        texture = self.get_layer_texture(bm.name, color)
                        # 确保UV (仅在需要时计算)
                        if texture is not None:
                            if not hasattr(main_mesh, 'active_t_coords') or main_mesh.active_t_coords is None:
                                # 为网格生成纹理坐标（平面映射）
                                try:
                                    main_mesh = main_mesh.texture_map_to_plane(inplace=False)
                                except Exception as e:
                                    self.log(f"警告: 为 {bm.name} 生成纹理坐标失败: {e}")
                                    texture = None

                            if side_mesh is not None and (not hasattr(side_mesh, 'active_t_coords') or side_mesh.active_t_coords is None):
                                try:
                                    side_mesh = side_mesh.texture_map_to_plane(inplace=False)
                                except Exception as e:
                                    self.log(f"警告: 为 {bm.name}_sides 生成纹理坐标失败: {e}")

                        actor = self.plotter.add_mesh(
                            main_mesh, texture=texture, color=color if texture is None else None,
                            opacity=opacity, smooth_shading=True, show_edges=show_edges,
                            edge_color='#000000', line_width=1, name=bm.name, ambient=0.3
                        )
                        if side_mesh:
                            self.plotter.add_mesh(
                                side_mesh, texture=texture, color=color if texture is None else None,
                                opacity=opacity, smooth_shading=False, lighting=False,
                                show_edges=show_edges, edge_color='#000000', line_width=1,
                                name=f"{bm.name}_sides"
                            )
                            self.plotter.actors[f"{bm.name}_sides"].SetVisibility(is_visible)
                    elif render_mode == '增强材质':
                        pbr_params = RockMaterial.get_pbr_params(bm.name)
                        actor = self.plotter.add_mesh(
                            main_mesh, color=color, opacity=opacity, smooth_shading=True,
                            pbr=True, metallic=pbr_params.get('metallic', 0.1),
                            roughness=pbr_params.get('roughness', 0.6),
                            diffuse=0.8, specular=0.5, show_edges=show_edges,
                            edge_color='#000000', line_width=1, name=bm.name
                        )
                        if side_mesh:
                            self.plotter.add_mesh(
                                side_mesh, color=color, opacity=opacity, smooth_shading=False,
                                lighting=False, show_edges=show_edges, edge_color='#000000',
                                line_width=1, name=f"{bm.name}_sides"
                            )
                            self.plotter.actors[f"{bm.name}_sides"].SetVisibility(is_visible)
                    else: # 基础渲染
                        actor = self.plotter.add_mesh(
                            main_mesh, color=color, opacity=opacity, smooth_shading=True,
                            show_edges=show_edges, edge_color='#000000', line_width=1,
                            name=bm.name, ambient=0.3
                        )
                        if side_mesh:
                            self.plotter.add_mesh(
                                side_mesh, color=color, opacity=opacity, smooth_shading=False,
                                lighting=False, show_edges=show_edges, edge_color='#000000',
                                line_width=1, name=f"{bm.name}_sides"
                            )
                            self.plotter.actors[f"{bm.name}_sides"].SetVisibility(is_visible)
                    
                    if actor:
                        actor.SetVisibility(is_visible)

            if hasattr(self, 'show_boreholes_cb') and self.show_boreholes_cb.isChecked():
                self.add_borehole_markers()

            self.update_contours()
            self.add_legend_safe(legend_entries)

            if hasattr(self, 'z_scale_slider'):
                self.plotter.set_scale(zscale=self.z_scale_slider.value() / 10.0)

            if camera_pos:
                self.plotter.camera_position = camera_pos
            else:
                self.plotter.reset_camera()
                self.plotter.view_isometric()

            if hasattr(self, 'show_boreholes_cb') and self.show_boreholes_cb.isChecked():
                self.plotter.enable_mesh_picking(
                    self.on_borehole_picked, show=False, show_message=False, left_clicking=True
                )

        except Exception as e:
            import traceback
            self.log(f"渲染失败: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.plotter.render_window.SetOffScreenRendering(0) # 恢复渲染
            self.is_rendering = False

    def add_borehole_markers(self):
        """添加钻孔位置标记"""
        if self.data_result is None or self.block_models is None:
            return

        try:
            coords = self.data_result['borehole_coords']
            borehole_ids = self.data_result['borehole_ids']

            # 计算模型整体高度范围
            z_max = max(bm.top_surface.max() for bm in self.block_models)
            z_min = min(bm.bottom_surface.min() for bm in self.block_models)
            height = z_max - z_min
            center_z = (z_max + z_min) / 2

            # 钻孔参数
            radius = 2.5  # 直径5m -> 半径2.5m
            
            for i, (x, y) in enumerate(coords):
                # 创建圆柱体
                cylinder = pv.Cylinder(
                    center=(x, y, center_z),
                    direction=(0, 0, 1),
                    radius=radius,
                    height=height,
                    resolution=20
                )
                
                # 添加钻孔ID到网格数据，用于拾取
                # 注意：PyVista的field_data需要是数组
                cylinder.field_data['borehole_id'] = [str(borehole_ids[i])]

                self.plotter.add_mesh(
                    cylinder,
                    color='#ff5555',
                    opacity=0.3, # 高透明度
                    smooth_shading=True,
                    name=f'borehole_cyl_{i}'
                )

                # 添加顶部标签
                self.plotter.add_point_labels(
                    [[x, y, z_max + 5]],
                    [borehole_ids[i]],
                    font_size=14,
                    text_color='#cdd6f4',
                    shape_color='#313244',
                    shape_opacity=0.8,
                    name=f'label_{i}'
                )

            # self.log(f"✓ 已添加 {len(coords)} 个钻孔标记")

        except Exception as e:
            self.log(f"添加钻孔标记失败: {str(e)}")

    def on_borehole_picked(self, mesh):
        """钻孔拾取回调"""
        if mesh is None:
            return
            
        # 检查是否有钻孔ID
        # PyVista的field_data通常是pyvista.DataSetAttributes
        if mesh.field_data and 'borehole_id' in mesh.field_data:
            try:
                # 获取ID
                bid_data = mesh.field_data['borehole_id']
                if len(bid_data) > 0:
                    borehole_id = bid_data[0]
                    # 如果是bytes类型(vtk有时会这样)，解码
                    if isinstance(borehole_id, bytes):
                        borehole_id = borehole_id.decode('utf-8')
                    self.show_borehole_details(borehole_id)
            except Exception as e:
                print(f"Pick error: {e}")

    def show_borehole_details(self, borehole_id):
        """显示钻孔详情"""
        if self.data_result is None or 'raw_df' not in self.data_result:
            return
            
        df = self.data_result['raw_df']
        # 筛选该钻孔的数据
        # 确保类型一致
        borehole_df = df[df['borehole_id'].astype(str) == str(borehole_id)].copy()
            
        if borehole_df.empty:
            self.log(f"未找到钻孔 {borehole_id} 的详细数据")
            return
            
        # 按深度排序
        if 'top_depth' in borehole_df.columns:
            borehole_df = borehole_df.sort_values('top_depth')
        elif 'layer_order' in borehole_df.columns:
            borehole_df = borehole_df.sort_values('layer_order')
        
        # 显示对话框
        dialog = BoreholeInfoDialog(borehole_id, borehole_df, self)
        dialog.exec()

    def select_all_layers(self):
        """全选地层"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if not item.isHidden():  # 只选择可见的项
                item.setCheckState(Qt.CheckState.Checked)
        self.update_layer_stats()

    def deselect_all_layers(self):
        """全不选地层"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if not item.isHidden():  # 只操作可见的项
                item.setCheckState(Qt.CheckState.Unchecked)
        self.update_layer_stats()

    def invert_layer_selection(self):
        """反选地层"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if not item.isHidden():  # 只操作可见的项
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
        self.update_layer_stats()

    def filter_layers(self, text):
        """过滤地层列表"""
        search_text = text.lower().strip()

        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            layer_name = item.text().lower()

            # 如果搜索文本为空或者匹配，显示该项
            if not search_text or search_text in layer_name:
                item.setHidden(False)
            else:
                item.setHidden(True)

        # 更新统计
        self.update_layer_stats()

    def update_layer_stats(self):
        """更新地层统计信息"""
        if not hasattr(self, 'layer_stats_label'):
            return

        total = self.layer_list.count()
        checked = 0
        visible = 0

        for i in range(total):
            item = self.layer_list.item(i)
            if not item.isHidden():
                visible += 1
                if item.checkState() == Qt.CheckState.Checked:
                    checked += 1

        # 更新标签
        if visible < total:
            self.layer_stats_label.setText(f"已选: {checked}/{visible} (共{total}层)")
        else:
            self.layer_stats_label.setText(f"已选: {checked}/{total}")

    def on_layer_item_changed(self, item):
        """地层勾选状态改变"""
        self.update_layer_stats()
        
        # 如果开启了剖面模式，需要清除合并网格缓存并重新渲染
        if hasattr(self, 'slice_cb') and self.slice_cb.isChecked():
            self.merged_mesh_cache = None
            self.request_render()
        else:
            # 标准模式下只更新可见性
            self.update_layer_visibility()

    def update_layer_visibility(self):
        """更新图层可见性和图例 - 优化版"""
        if not self.plotter or not self.block_models:
            return

        # 获取所有勾选的层
        visible_layers = set()
        if hasattr(self, 'layer_list'):
            for i in range(self.layer_list.count()):
                item = self.layer_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    visible_layers.add(item.text())
        
        legend_entries = []
        
        # 批量更新可见性，避免重建整个场景
        self.plotter.render_window.SetOffScreenRendering(1) # 暂停渲染
        try:
            for i, bm in enumerate(self.block_models):
                actor_name = bm.name
                is_visible = bm.name in visible_layers
                
                # 主网格
                if actor_name in self.plotter.actors:
                    actor = self.plotter.actors[actor_name]
                    if actor.GetVisibility() != is_visible:
                        actor.SetVisibility(is_visible)
                
                # 侧面网格
                side_actor_name = f"{bm.name}_sides"
                if side_actor_name in self.plotter.actors:
                    side_actor = self.plotter.actors[side_actor_name]
                    if side_actor.GetVisibility() != is_visible:
                        side_actor.SetVisibility(is_visible)
                
                # 收集图例
                if is_visible:
                    # 从缓存或材质获取颜色
                    color = RockMaterial.get_color(bm.name, i)
                    legend_entries.append((bm.name, color))
            
            # 更新图例
            self.add_legend_safe(legend_entries)
            
        finally:
            self.plotter.render_window.SetOffScreenRendering(0) # 恢复渲染
            self.plotter.render() # 触发一次重绘

    def show_layer_context_menu(self, position):
        """显示地层列表右键菜单"""
        item = self.layer_list.itemAt(position)
        if not item:
            return
            
        layer_name = item.text()
        menu = QMenu()
        
        # 仅显示此层
        action_solo = QAction(f"仅显示 '{layer_name}'", self)
        action_solo.triggered.connect(lambda: self.solo_layer(layer_name))
        menu.addAction(action_solo)
        
        # 定位到此层
        action_focus = QAction("聚焦到此层", self)
        action_focus.triggered.connect(lambda: self.focus_layer(layer_name))
        menu.addAction(action_focus)
        
        menu.addSeparator()
        
        # 属性
        action_props = QAction("查看属性...", self)
        action_props.triggered.connect(lambda: self.show_layer_properties(layer_name))
        menu.addAction(action_props)
        
        menu.exec(self.layer_list.mapToGlobal(position))

    def solo_layer(self, target_layer):
        """仅显示指定地层"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.text() == target_layer:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.update_layer_stats()

    def focus_layer(self, layer_name):
        """聚焦到指定地层"""
        if not self.plotter or layer_name not in self.plotter.actors:
            return
            
        actor = self.plotter.actors[layer_name]
        if not actor.GetVisibility():
            # 如果不可见，先显示
            items = self.layer_list.findItems(layer_name, Qt.MatchFlag.MatchExactly)
            if items:
                items[0].setCheckState(Qt.CheckState.Checked)
                self.update_layer_visibility()
        
        # 获取包围盒并聚焦
        bounds = actor.GetBounds()
        if bounds:
            center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2)
            # 简单的聚焦逻辑：移动相机到中心点附近
            self.plotter.camera.focal_point = center
            # 保持当前视角方向，但调整距离
            dist = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 2.0
            pos = self.plotter.camera.position
            foc = self.plotter.camera.focal_point
            vec = np.array(pos) - np.array(foc)
            vec = vec / np.linalg.norm(vec) * dist
            self.plotter.camera.position = tuple(np.array(foc) + vec)
            self.plotter.render()

    def show_layer_properties(self, layer_name):
        """显示地层属性"""
        if not self.block_models:
            return
            
        target_bm = None
        for bm in self.block_models:
            if bm.name == layer_name:
                target_bm = bm
                break
                
        if not target_bm:
            return
            
        # 计算统计信息
        thickness = target_bm.top_surface - target_bm.bottom_surface
        avg_thick = np.nanmean(thickness)
        max_thick = np.nanmax(thickness)
        min_thick = np.nanmin(thickness)
        # 简单体积估算
        dx = self.XI[0,1]-self.XI[0,0]
        dy = self.YI[1,0]-self.YI[0,0]
        volume = np.nansum(thickness) * dx * dy
        
        msg = f"""
        <h3>地层: {layer_name}</h3>
        <hr>
        <b>厚度统计:</b><br>
        平均: {avg_thick:.2f} m<br>
        最大: {max_thick:.2f} m<br>
        最小: {min_thick:.2f} m<br>
        <br>
        <b>体积估算:</b><br>
        {volume/10000:.2f} 万 m³
        """
        QMessageBox.information(self, f"属性 - {layer_name}", msg)

    def on_edges_toggled(self):
        """网格显示切换 - 轻量级更新"""
        if not self.plotter or not self.block_models:
            return
        
        show_edges = self.show_edges_cb.isChecked()
        
        # 直接更新所有Actor的边缘显示属性，无需重建场景
        try:
            for bm in self.block_models:
                actor_name = bm.name
                if actor_name in self.plotter.actors:
                    actor = self.plotter.actors[actor_name]
                    if hasattr(actor, 'prop'):
                        actor.prop.show_edges = show_edges
                        actor.prop.edge_color = (0, 0, 0)  # 黑色边缘
                
                # 更新侧面
                side_actor_name = f"{bm.name}_sides"
                if side_actor_name in self.plotter.actors:
                    actor = self.plotter.actors[side_actor_name]
                    if hasattr(actor, 'prop'):
                        actor.prop.show_edges = show_edges
                        actor.prop.edge_color = (0, 0, 0)
            
            self.plotter.render()
        except Exception as e:
            self.log(f"切换网格显示失败: {e}")
            # 回退到完整重建
            self.request_render()

    def on_render_mode_changed(self, mode: str):
        """渲染模式改变"""
        if self.block_models is not None:
            self.request_render()

    def on_opacity_changed(self, value: int):
        """透明度改变 - 实时更新"""
        opacity = value / 100.0
        self.opacity_label.setText(f"{opacity:.2f}")
        
        if not self.plotter or not self.block_models:
            return

        # 直接更新所有层Actor的透明度
        for bm in self.block_models:
            actor_name = bm.name
            if actor_name in self.plotter.actors:
                actor = self.plotter.actors[actor_name]
                if hasattr(actor, 'prop'):
                    actor.prop.opacity = opacity
            
            # 更新侧面透明度
            side_actor_name = f"{bm.name}_sides"
            if side_actor_name in self.plotter.actors:
                actor = self.plotter.actors[side_actor_name]
                if hasattr(actor, 'prop'):
                    actor.prop.opacity = opacity
        
        self.plotter.render()

    def on_sides_toggled(self):
        """侧面显示切换 - 轻量级更新"""
        if not self.plotter or not self.block_models:
            return
        
        show_sides = self.show_sides_cb.isChecked()
        
        # 检查是否有侧面Actor存在
        has_side_actors = any(
            f"{bm.name}_sides" in self.plotter.actors 
            for bm in self.block_models
        )
        
        if has_side_actors:
            # 侧面Actor已存在，只切换可见性
            for bm in self.block_models:
                side_actor_name = f"{bm.name}_sides"
                if side_actor_name in self.plotter.actors:
                    actor = self.plotter.actors[side_actor_name]
                    actor.SetVisibility(show_sides)
            self.plotter.render()
        else:
            # 侧面Actor不存在，需要重建（只在开启侧面时）
            if show_sides:
                self.request_render()

    def on_boreholes_toggled(self):
        """钻孔显示切换 - 实时"""
        if not self.plotter or not self.block_models:
            return
            
        if self.show_boreholes_cb.isChecked():
            self.add_borehole_markers()
        else:
            # 移除钻孔标记
            if self.data_result:
                 n_boreholes = len(self.data_result['borehole_ids'])
                 for i in range(n_boreholes):
                     self.plotter.remove_actor(f'borehole_cyl_{i}')
                     self.plotter.remove_actor(f'label_{i}')

    def refresh_render(self):
        """刷新渲染 - 强制完整重建"""
        if self.block_models is not None and PYVISTA_AVAILABLE and self.plotter is not None:
            self.render_3d_model()

    def export_model(self, format_type: str):
        """导出模型"""
        if self.block_models is None:
            QMessageBox.warning(self, "警告", "请先构建三维模型!")
            return

        if format_type == 'png':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存PNG截图", "geological_model.png", "PNG Files (*.png)"
            )
        elif format_type == 'html':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存HTML", "geological_model.html", "HTML Files (*.html)"
            )
        elif format_type == 'obj':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存OBJ", "geological_model.obj", "OBJ Files (*.obj)"
            )
        elif format_type == 'stl':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存STL", "geological_model.stl", "STL Files (*.stl)"
            )
        elif format_type == 'vtk':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存VTK", "geological_model.vtk", "VTK Files (*.vtk)"
            )
        elif format_type == 'flac3d':
            # 根据格式选择确定文件扩展名
            format_idx = self.flac3d_format_combo.currentIndex() if hasattr(self, 'flac3d_format_combo') else 0
            if format_idx == 0:  # f3grid
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存FLAC3D网格", "geological_model.f3grid", "FLAC3D Grid Files (*.f3grid)"
                )
            elif format_idx == 1:  # FPN
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存FPN网格", "geological_model.fpn", "Midas GTS NX FPN Files (*.fpn)"
                )
            else:  # f3dat 脚本
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存FLAC3D脚本", "geological_model.f3dat", "FLAC3D Files (*.f3dat)"
                )
        else:
            return

        if not file_path:
            return

        self.log(f"\n正在导出 {format_type.upper()}...")

        try:
            if format_type == 'png' and self.plotter:
                self.plotter.screenshot(file_path, scale=2)
            elif format_type == 'html' and self.plotter:
                self.plotter.export_html(file_path)
            elif format_type in ['obj', 'stl', 'vtk']:
                renderer = GeologicalModelRenderer()

                selected_layers = set()
                if hasattr(self, 'layer_list'):
                    for i in range(self.layer_list.count()):
                        item = self.layer_list.item(i)
                        if item.checkState() == Qt.CheckState.Checked:
                            selected_layers.add(item.text())
                else:
                    selected_layers = {bm.name for bm in self.block_models}

                show_sides = self.show_sides_cb.isChecked() if hasattr(self, 'show_sides_cb') else True

                for i, bm in enumerate(self.block_models):
                    if bm.name not in selected_layers:
                        continue
                    mesh = renderer.create_layer_mesh(
                        self.XI, self.YI,
                        bm.top_surface, bm.bottom_surface,
                        bm.name,
                        add_sides=show_sides
                    )
                    renderer.meshes.append(mesh)
                renderer.export_mesh(file_path, file_format=format_type)

            elif format_type == 'flac3d':
                if not FLAC3D_EXPORTER_AVAILABLE:
                    QMessageBox.warning(self, "警告", "FLAC3D导出器不可用!\n请检查 src/exporters/flac3d_enhanced_exporter.py")
                    return

                # 准备FLAC3D导出数据
                self.log("准备FLAC3D导出数据...")

                # 获取选中的层
                selected_layers = set()
                if hasattr(self, 'layer_list'):
                    for i in range(self.layer_list.count()):
                        item = self.layer_list.item(i)
                        if item.checkState() == Qt.CheckState.Checked:
                            selected_layers.add(item.text())
                else:
                    selected_layers = {bm.name for bm in self.block_models}

                # 构建层数据（FLAC3D格式）
                layers_data = []
                for i, bm in enumerate(self.block_models):
                    if bm.name not in selected_layers:
                        continue

                    # 从2D网格创建1D坐标
                    ny, nx = self.XI.shape
                    x = self.XI[0, :]
                    y = self.YI[:, 0]

                    layer_dict = {
                        'name': bm.name,
                        'grid_x': x,
                        'grid_y': y,
                        'top_surface_z': bm.top_surface,
                        'bottom_surface_z': bm.bottom_surface,
                        'properties': {
                            'density': 2400,  # 默认密度
                            'youngs_modulus': 10e9,  # 默认杨氏模量
                            'poisson_ratio': 0.25,  # 默认泊松比
                            'cohesion': 2e6,  # 默认内聚力
                            'friction_angle': 30  # 默认摩擦角
                        }
                    }
                    layers_data.append(layer_dict)

                if not layers_data:
                    QMessageBox.warning(self, "警告", "没有选中的地层可导出!")
                    return

                # 估算网格大小并给出建议
                total_cells = sum([(len(ld['grid_x'])-1) * (len(ld['grid_y'])-1) for ld in layers_data])
                downsample = self.flac3d_downsample_spin.value() if hasattr(self, 'flac3d_downsample_spin') else 1
                estimated_cells = total_cells // (downsample * downsample)

                # 大模型警告
                if estimated_cells > 100000:
                    reply = QMessageBox.question(
                        self, "大模型警告",
                        f"预计生成 {estimated_cells:,} 个单元，文件可能很大且FLAC3D加载缓慢!\n\n"
                        f"建议:\n"
                        f"- 当前降采样: {downsample}x\n"
                        f"- 推荐降采样: {max(2, downsample)}x 或更高\n"
                        f"- 或减少选中的地层数量\n\n"
                        f"是否继续当前设置?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return

                # 选择导出器格式
                # 0: f3grid (推荐), 1: FPN (中间格式), 2: 紧凑脚本, 3: 完整脚本
                format_idx = 0
                if hasattr(self, 'flac3d_format_combo'):
                    format_idx = self.flac3d_format_combo.currentIndex()

                format_names = ['f3grid', 'FPN', '紧凑脚本', '完整脚本']
                # 创建导出器并导出
                self.log(f"导出 {len(layers_data)} 个地层到FLAC3D...")
                self.log(f"降采样因子: {downsample}x (网格减少 {100*(1-1/(downsample*downsample)):.0f}%)")
                self.log(f"格式: {format_names[format_idx]}")

                if format_idx == 0:  # f3grid 格式
                    if not F3GRID_V2_AVAILABLE:
                        QMessageBox.warning(self, "警告", "F3Grid导出器不可用!\n请检查 src/exporters/f3grid_exporter_v2.py")
                        return

                    # 识别所有煤层
                    coal_layer_indices = []
                    coal_layer_names = []
                    for i, layer_dict in enumerate(layers_data):
                        name = layer_dict['name']
                        if '煤' in name or 'coal' in name.lower():
                            coal_layer_indices.append(i)
                            coal_layer_names.append(f"[{i}] {name}")

                    # 如果有多个煤层，让用户选择
                    selected_coal_indices = None
                    if len(coal_layer_indices) > 3:  # 超过3个煤层才询问
                        dialog = QDialog(self)
                        dialog.setWindowTitle("选择高密度煤层")
                        dialog.setMinimumWidth(500)
                        layout = QVBoxLayout(dialog)

                        # 说明文字
                        label = QLabel(
                            f"识别到 {len(coal_layer_indices)} 个煤层。\n\n"
                            f"为了优化性能，请选择需要使用高密度网格的煤层。\n"
                            f"未选中的煤层将使用常规降采样率（{downsample}x）。\n"
                            f"选中的煤层及其上下2层将使用原始密度（1x）。"
                        )
                        label.setWordWrap(True)
                        layout.addWidget(label)

                        # 煤层列表（多选）
                        list_widget = QListWidget()
                        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
                        for name in coal_layer_names:
                            list_widget.addItem(name)
                        # 默认全选
                        for i in range(list_widget.count()):
                            list_widget.item(i).setSelected(True)
                        layout.addWidget(list_widget)

                        # 全选/全不选按钮
                        btn_layout = QHBoxLayout()
                        select_all_btn = QPushButton("全选")
                        select_none_btn = QPushButton("全不选")
                        select_all_btn.clicked.connect(lambda: list_widget.selectAll())
                        select_none_btn.clicked.connect(lambda: list_widget.clearSelection())
                        btn_layout.addWidget(select_all_btn)
                        btn_layout.addWidget(select_none_btn)
                        btn_layout.addStretch()
                        layout.addLayout(btn_layout)

                        # 确认/取消按钮
                        button_box = QDialogButtonBox(
                            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                        )
                        button_box.accepted.connect(dialog.accept)
                        button_box.rejected.connect(dialog.reject)
                        layout.addWidget(button_box)

                        if dialog.exec() == QDialog.DialogCode.Accepted:
                            # 获取选中的煤层索引
                            selected_items = list_widget.selectedItems()
                            if selected_items:
                                selected_coal_indices = []
                                for item in selected_items:
                                    # 从 "[index] name" 格式中提取index
                                    text = item.text()
                                    idx = int(text.split(']')[0][1:])
                                    selected_coal_indices.append(idx)
                                    self.log(f"  选中煤层: [{idx}] {text.split(']')[1].strip()}")
                                self.log(f"用户选择 {len(selected_coal_indices)} 个煤层使用高密度网格: {selected_coal_indices}")
                            else:
                                self.log("未选择任何煤层，所有地层使用统一降采样")
                                selected_coal_indices = []  # 空列表表示没有高密度煤层
                        else:
                            self.log("取消导出")
                            return
                    else:
                        # 煤层数量少，使用所有煤层
                        if coal_layer_indices:
                            self.log(f"煤层数量 ≤ 3，自动对所有煤层使用高密度网格")

                    exporter = F3GridExporterV2()

                    # 获取接触面选项
                    create_interfaces = self.create_interfaces_checkbox.isChecked() if hasattr(self, 'create_interfaces_checkbox') else False

                    export_options = {
                        'downsample_factor': downsample,
                        'coal_downsample_factor': 1,  # 煤层区域使用1x（原始密度）
                        'coal_adjacent_layers': 2,  # 煤层上下各2层使用高密度
                        'selected_coal_layers': selected_coal_indices,  # 用户选择的煤层
                        'min_zone_thickness': 0.001,
                        'coord_precision': 6,
                        'check_overlap': True,
                        'create_interfaces': create_interfaces  # 接触面模式
                    }

                    exporter.export(
                        data={'layers': layers_data},
                        output_path=file_path,
                        options=export_options
                    )

                    self.log(f"FLAC3D导出统计:")
                    self.log(f"  总节点数: {exporter.stats.total_gridpoints}")
                    self.log(f"  共享节点数: {exporter.stats.shared_nodes}")
                    self.log(f"  总单元数: {exporter.stats.total_zones}")
                    if exporter.stats.min_thickness < float('inf'):
                        self.log(f"  厚度范围: {exporter.stats.min_thickness:.3f}m - {exporter.stats.max_thickness:.3f}m")
                    # 显示坐标范围
                    x_min, x_max = exporter.stats.coord_range_x
                    y_min, y_max = exporter.stats.coord_range_y
                    z_min, z_max = exporter.stats.coord_range_z
                    sx, sy, sz = exporter.stats.model_size
                    if sx > 0 or sy > 0 or sz > 0:
                        self.log(f"  原始坐标范围:")
                        self.log(f"    X: [{x_min:.2f}, {x_max:.2f}] (尺寸: {sx:.2f}m)")
                        self.log(f"    Y: [{y_min:.2f}, {y_max:.2f}] (尺寸: {sy:.2f}m)")
                        self.log(f"    Z: [{z_min:.2f}, {z_max:.2f}] (尺寸: {sz:.2f}m)")
                    ox, oy, oz = exporter.stats.origin_offset
                    if ox != 0 or oy != 0 or oz != 0:
                        self.log(f"  坐标系统: 相对坐标")
                        self.log(f"  原点偏移: X={ox:.2f}m, Y={oy:.2f}m, Z={oz:.2f}m")
                    self.log(f"\n在FLAC3D中导入:")
                    self.log(f'  zone import f3grid "{os.path.basename(file_path)}"')

                elif format_idx == 1:  # FPN 格式
                    if not FPN_EXPORTER_AVAILABLE:
                        QMessageBox.warning(self, "警告", "FPN导出器不可用!\n请检查 src/exporters/fpn_exporter.py")
                        return

                    # FPN 格式也支持煤层选择
                    coal_layer_indices = []
                    coal_layer_names = []
                    for i, layer_dict in enumerate(layers_data):
                        name = layer_dict['name']
                        if '煤' in name or 'coal' in name.lower():
                            coal_layer_indices.append(i)
                            coal_layer_names.append(f"[{i}] {name}")

                    # 如果有多个煤层，让用户选择（与f3grid相同逻辑）
                    selected_coal_indices = None
                    if len(coal_layer_indices) > 3:  # 超过3个煤层才询问
                        dialog = QDialog(self)
                        dialog.setWindowTitle("选择高密度煤层")
                        dialog.setMinimumWidth(500)
                        layout = QVBoxLayout(dialog)

                        # 说明文字
                        label = QLabel(
                            f"识别到 {len(coal_layer_indices)} 个煤层。\n\n"
                            f"为了优化性能，请选择需要使用高密度网格的煤层。\n"
                            f"未选中的煤层将使用常规降采样率（{downsample}x）。\n"
                            f"选中的煤层及其上下2层将使用原始密度（1x）。"
                        )
                        label.setWordWrap(True)
                        layout.addWidget(label)

                        # 煤层列表（多选）
                        list_widget = QListWidget()
                        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
                        for name in coal_layer_names:
                            list_widget.addItem(name)
                        # 默认全选
                        for i in range(list_widget.count()):
                            list_widget.item(i).setSelected(True)
                        layout.addWidget(list_widget)

                        # 全选/全不选按钮
                        btn_layout = QHBoxLayout()
                        select_all_btn = QPushButton("全选")
                        select_none_btn = QPushButton("全不选")
                        select_all_btn.clicked.connect(lambda: list_widget.selectAll())
                        select_none_btn.clicked.connect(lambda: list_widget.clearSelection())
                        btn_layout.addWidget(select_all_btn)
                        btn_layout.addWidget(select_none_btn)
                        btn_layout.addStretch()
                        layout.addLayout(btn_layout)

                        # 确认/取消按钮
                        button_box = QDialogButtonBox(
                            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                        )
                        button_box.accepted.connect(dialog.accept)
                        button_box.rejected.connect(dialog.reject)
                        layout.addWidget(button_box)

                        if dialog.exec() == QDialog.DialogCode.Accepted:
                            # 获取选中的煤层索引
                            selected_items = list_widget.selectedItems()
                            if selected_items:
                                selected_coal_indices = []
                                for item in selected_items:
                                    # 从 "[index] name" 格式中提取index
                                    text = item.text()
                                    idx = int(text.split(']')[0][1:])
                                    selected_coal_indices.append(idx)
                                    self.log(f"  选中煤层: [{idx}] {text.split(']')[1].strip()}")
                                self.log(f"用户选择 {len(selected_coal_indices)} 个煤层使用高密度网格: {selected_coal_indices}")
                            else:
                                self.log("未选择任何煤层，所有地层使用统一降采样")
                                selected_coal_indices = []  # 空列表表示没有高密度煤层
                        else:
                            self.log("取消导出")
                            return
                    else:
                        # 煤层数量少，使用所有煤层
                        if coal_layer_indices:
                            self.log(f"煤层数量 ≤ 3，自动对所有煤层使用高密度网格")

                    exporter = FPNExporter()

                    # 获取接触面选项
                    create_interfaces = self.create_interfaces_checkbox.isChecked() if hasattr(self, 'create_interfaces_checkbox') else False

                    export_options = {
                        'downsample_factor': downsample,
                        'coal_downsample_factor': 1,
                        'coal_adjacent_layers': 2,
                        'selected_coal_layers': selected_coal_indices,
                        'create_interfaces': create_interfaces  # 接触面模式
                    }

                    exporter.export(
                        data={'layers': layers_data},
                        output_path=file_path,
                        options=export_options
                    )

                    self.log(f"✓ FPN导出成功!")
                    self.log(f"提示: FPN文件可使用Midas转换工具转换为FLAC3D f3grid格式")

                elif format_idx == 2:  # 紧凑脚本
                    exporter = CompactFLAC3DExporter()
                    export_options = {
                        'downsample_factor': downsample,
                        'normalize_coords': False,
                        'validate_mesh': True,
                        'coord_precision': 3
                    }
                    exporter.export(
                        data={'layers': layers_data, 'title': 'GNN地质建模系统', 'author': 'PyQt6版'},
                        output_path=file_path,
                        options=export_options
                    )
                    self.log(f"FLAC3D导出统计:")
                    self.log(f"  总节点数: {exporter.stats['total_nodes']}")
                    self.log(f"  共享节点数: {exporter.stats['shared_nodes']}")
                    self.log(f"  总单元数: {exporter.stats['total_zones']}")
                    self.log(f"  厚度范围: {exporter.stats['min_thickness']:.2f}m - {exporter.stats['max_thickness']:.2f}m")

                else:  # 完整脚本
                    exporter = EnhancedFLAC3DExporter()
                    export_options = {
                        'downsample_factor': downsample,
                        'normalize_coords': False,
                        'validate_mesh': True,
                        'coord_precision': 3
                    }
                    exporter.export(
                        data={'layers': layers_data, 'title': 'GNN地质建模系统', 'author': 'PyQt6版'},
                        output_path=file_path,
                        options=export_options
                    )
                    self.log(f"FLAC3D导出统计:")
                    self.log(f"  总节点数: {exporter.stats['total_nodes']}")
                    self.log(f"  共享节点数: {exporter.stats['shared_nodes']}")
                    self.log(f"  总单元数: {exporter.stats['total_zones']}")
                    self.log(f"  厚度范围: {exporter.stats['min_thickness']:.2f}m - {exporter.stats['max_thickness']:.2f}m")

            self.log(f"✓ 导出成功: {file_path}")
            
            # 询问是否打开文件夹
            reply = QMessageBox.question(
                self, "导出成功", 
                f"文件已保存:\n{file_path}\n\n是否打开所在文件夹?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                folder_path = os.path.dirname(file_path)
                try:
                    os.startfile(folder_path)
                except Exception as e:
                    self.log(f"无法打开文件夹: {e}")

        except Exception as e:
            import traceback
            error_msg = f"导出失败: {str(e)}\n{traceback.format_exc()}"
            self.log(f"✗ {error_msg}")
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    def on_error(self, message: str):
        """错误处理"""
        self.log(f"\n✗ 错误: {message}")

        self.set_busy_state(False)
        self.train_btn.setEnabled(True if self.data_result else False)
        self.model_btn.setEnabled(True if self.predictor else False)
        self.progress_bar.setVisible(False)

        # 关闭进度对话框
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_message("✗ 错误")
            self.progress_dialog.set_detail(message[:100])  # 显示前100个字符
            self.progress_dialog.auto_close_on_complete()

        QMessageBox.critical(self, "错误", message)

    def _on_training_progress(self, percent: int):
        """训练进度更新回调"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_progress(percent)
            # 可以根据阶段更新消息
            if percent < 30:
                self.progress_dialog.set_message("正在拟合模型...")
            elif percent < 80:
                self.progress_dialog.set_message("正在评估性能...")
            else:
                self.progress_dialog.set_message("正在完成训练...")

    def _on_modeling_progress(self, percent: int):
        """建模进度更新回调"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.set_progress(percent)
            # 根据进度更新消息
            if percent < 20:
                self.progress_dialog.set_message("正在生成网格...")
            elif percent < 60:
                self.progress_dialog.set_message("正在预测厚度...")
            else:
                self.progress_dialog.set_message("正在构建三维模型...")


def main():
    # Set global exception hook
    sys.excepthook = global_exception_hook
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # High DPI support
    # Note: PyQt6 enables high DPI by default, but we set these for compatibility if using PyQt5 or specific environments
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    # Set application icon if available
    # app.setWindowIcon(QIcon('resources/icon.ico'))

    window = GeologicalModelingApp()
    window.showMaximized() # 默认最大化启动
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
