# FLAC3D导出功能使用指南

## 功能概述

PyQt6版本已集成FLAC3D增强导出功能，确保层间节点共享，实现应力和位移的正确传导。

## 核心特性

### 1. 层间节点共享 ✓
- 上层底面节点 = 下层顶面节点（完全共享）
- 确保应力/位移在层间正确传导
- 避免层间出现空隙或重叠

### 2. 正确的FLAC3D命令语法 ✓
- 兼容FLAC3D 7.0+版本
- 使用标准的`zone gridpoint create`和`zone create brick`命令
- 自动生成分组命令

### 3. 网格质量验证 ✓
- 自动检测负体积单元
- 自动修正节点顺序
- 输出详细统计信息

## 在PyQt6应用中使用

### 步骤1：构建三维模型

1. 加载钻孔数据
2. 训练模型（传统方法或GNN）
3. 点击"🏗️ 构建三维模型"

### 步骤2：选择要导出的地层

在左侧"渲染控制"面板的"显示地层"列表中：
- 使用Ctrl+点击多选地层
- 只有选中的地层会被导出到FLAC3D

### 步骤3：导出FLAC3D网格

1. 点击绿色的"**FLAC3D网格**"按钮
2. 选择保存位置（建议使用`.f3dat`扩展名）
3. 等待导出完成
4. 查看控制台日志中的统计信息

### 步骤4：在FLAC3D中导入

在FLAC3D 7.0+中执行：

```fish
; 导入网格
program call "geological_model.f3dat"

; 检查模型
zone list information
```

## 导出统计信息说明

导出完成后，控制台会显示：

```
FLAC3D导出统计:
  总节点数: 400              # 实际创建的节点总数
  共享节点数: 1544           # 节点被引用的总次数
  总单元数: 243              # 生成的单元总数
  厚度范围: 2.50m - 12.80m  # 地层厚度范围
```

**节点共享效率**：
- 理想情况：每个单元8个节点，243个单元 = 1944个节点引用
- 实际：1544个共享 / 1944个总引用 = 79.4%共享率
- 说明层间接触面的节点已正确共享

## 材料属性设置

当前版本使用默认材料属性，可以在FLAC3D中手动设置：

```fish
; 设置材料属性（示例）
zone cmodel assign mohr-coulomb

; 泥岩
zone property density=2400 shear=4e9 bulk=6.67e9 range group '泥岩'
zone property cohesion=2e6 friction=30 range group '泥岩'

; 煤层
zone property density=1400 shear=0.77e9 bulk=1.33e9 range group '煤层'
zone property cohesion=1e6 friction=25 range group '煤层'

; 砂岩
zone property density=2600 shear=6e9 bulk=10e9 range group '砂岩'
zone property cohesion=5e6 friction=35 range group '砂岩'
```

## 完整FLAC3D分析脚本示例

```fish
; ==========================================
; FLAC3D 完整分析脚本
; ==========================================

; 1. 导入模型
program call "geological_model.f3dat"

; 2. 设置本构模型
zone cmodel assign mohr-coulomb

; 3. 设置材料属性
; （参考上面的材料属性设置）

; 4. 边界条件
zone face apply velocity-x 0 range position-x 0
zone face apply velocity-x 0 range position-x 100
zone face apply velocity-y 0 range position-y 0
zone face apply velocity-y 0 range position-y 100
zone face apply velocity-z 0 range position-z 0

; 5. 重力初始化
model gravity 0 0 -9.81
zone initialize-stresses ratio 0.5

; 6. 求解到平衡
model solve ratio-average 1e-5
model save "initial_equilibrium.f3sav"

; 7. 检查结果
zone list displacement
zone list stress
```

## 验证层间连续性

在FLAC3D中验证节点共享：

```fish
; 列出所有节点信息
zone gridpoint list

; 检查特定Z坐标的节点（层间接触面）
zone gridpoint list range position-z 10.0 10.1

; 检查单元连接性
zone list information
```

## 性能优化建议

### 降低网格分辨率
- 在"建模配置"中调整"网格分辨率"
- 推荐值：20-50（快速预览）
- 生产模型：50-100

### 选择性导出
- 只导出需要分析的地层
- 减少不必要的单元数量

### 降采样导出
目前版本默认不降采样，如需降采样可以在构建模型时使用较低分辨率。

## 常见问题

### Q1: 导出的模型在FLAC3D中层间不连续？
**A**: 这是之前版本的问题。增强版导出器已完全解决，确保上层底面 = 下层顶面。

### Q2: 如何检查节点是否真正共享？
**A**: 在FLAC3D中：
```fish
zone gridpoint list range position-z [接触面Z坐标]
```
如果同一Z坐标有多个单元引用同一节点ID，说明共享成功。

### Q3: 负体积单元是什么？
**A**: 节点顺序错误导致的"内翻"单元。增强版导出器会自动检测并修复。

### Q4: 可以导出部分地层吗？
**A**: 可以！在导出前在"显示地层"列表中选择要导出的地层。

### Q5: 导出的文件很大怎么办？
**A**:
- 降低建模分辨率
- 只导出必要的地层
- 使用合理的网格密度

## 技术支持

如遇问题，请检查：
1. 是否已构建三维模型
2. 是否选择了要导出的地层
3. 查看控制台日志中的详细信息
4. 检查导出的`.f3dat`文件是否生成

---

**版本**: v2.0 增强版
**更新日期**: 2025-12-17
**集成到**: PyQt6高性能应用
