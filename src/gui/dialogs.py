from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView
)

class BoreholeInfoDialog(QDialog):
    """显示钻孔详细信息的对话框"""
    def __init__(self, borehole_id, df_layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"钻孔详情: {borehole_id}")
        self.resize(600, 400)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QTableWidget { 
                background-color: #181825; 
                color: #cdd6f4; 
                gridline-color: #45475a;
                border: 1px solid #45475a;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #cdd6f4;
                padding: 4px;
                border: 1px solid #45475a;
            }
            QTableWidget::item:selected { background-color: #45475a; }
        """)

        layout = QVBoxLayout(self)

        # 标题信息
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"<h3>钻孔编号: {borehole_id}</h3>"))
        
        # 计算总深度
        total_depth = df_layers['bottom_depth'].max() if not df_layers.empty else 0
        info_layout.addWidget(QLabel(f"总深度: {total_depth:.2f} m"))
        
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["层序", "地层名称", "岩性", "厚度(m)", "底板深度(m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        
        # 填充数据
        self.table.setRowCount(len(df_layers))
        for i, (_, row) in enumerate(df_layers.iterrows()):
            self.table.setItem(i, 0, QTableWidgetItem(str(row.get('layer_order', i+1))))
            self.table.setItem(i, 1, QTableWidgetItem(str(row.get('layer_name', ''))))
            self.table.setItem(i, 2, QTableWidgetItem(str(row.get('lithology', ''))))
            self.table.setItem(i, 3, QTableWidgetItem(f"{row.get('thickness', 0):.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{row.get('bottom_depth', 0):.2f}"))

        layout.addWidget(self.table)
