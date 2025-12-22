# FLAC3D f3grid 导出器 - B8 节点顺序修复

## 修复日期
2025-12-21

## 问题描述

根据 `优化建议.md` 的指导，发现 f3grid_exporter_v2.py 中的 B8 单元节点顺序不符合 FLAC3D 官方标准，导致：
- **错误**：`Zone 1 has a negative volume`
- **错误**：`tet volumes are <= 0`
- **原因**：节点顺序错误导致 FLAC3D 在内部构造四面体时出现负体积/零体积

## 根本原因

### 错误的旧顺序（已修复）
```python
# ❌ 错误的顺序：先底面4个，后顶面4个
gp_ids = [
    gp_b_00,  # 底面 SW
    gp_b_10,  # 底面 SE
    gp_b_11,  # 底面 NE
    gp_b_01,  # 底面 NW
    gp_t_00,  # 顶面 SW
    gp_t_10,  # 顶面 SE
    gp_t_11,  # 顶面 NE
    gp_t_01,  # 顶面 NW
]
```

**问题**：这种"先底面后顶面"的顺序不符合 FLAC3D 官方的 B8 标准。

### 正确的新顺序（已修复）
```python
# ✅ 正确的顺序：FLAC3D 官方标准（交织模式）
# 参考: docs.itascacg.com - "Orientation of Nodes and Faces within a Zone"
gp_ids = [
    gp_b_00,  # 1: SW_bot (底面西南, x_min, y_min)
    gp_b_10,  # 2: SE_bot (底面东南, x_max, y_min)
    gp_b_01,  # 3: NW_bot (底面西北, x_min, y_max)  ← 关键修正
    gp_t_00,  # 4: SW_top (顶面西南)  ← 关键修正
    gp_b_11,  # 5: NE_bot (底面东北, x_max, y_max)  ← 关键修正
    gp_t_01,  # 6: NW_top (顶面西北)  ← 关键修正
    gp_t_10,  # 7: SE_top (顶面东南)  ← 关键修正
    gp_t_11,  # 8: NE_top (顶面东北)
]
```

**关键理解**：FLAC3D 使用的是**交织模式**，不是"先底面后顶面"的分组模式！

## 修复详情

### 1. 修正 B8 单元的 gp_ids 顺序
**文件**: `src/exporters/f3grid_exporter_v2.py`
**位置**: 第 560-569 行

**修改前**：
```python
gp_ids = [
    gp_b_00, gp_b_10, gp_b_11, gp_b_01,  # 底面 4 个
    gp_t_00, gp_t_10, gp_t_11, gp_t_01   # 顶面 4 个
]
```

**修改后**：
```python
gp_ids = [
    gp_b_00,  # 1: SW_bot
    gp_b_10,  # 2: SE_bot
    gp_b_01,  # 3: NW_bot  ← 交织开始
    gp_t_00,  # 4: SW_top  ← 插入顶面
    gp_b_11,  # 5: NE_bot  ← 回到底面
    gp_t_01,  # 6: NW_top  ← 插入顶面
    gp_t_10,  # 7: SE_top  ← 插入顶面
    gp_t_11,  # 8: NE_top
]
```

### 2. 修正四面体体积检查的坐标数组
**文件**: `src/exporters/f3grid_exporter_v2.py`
**位置**: 第 573-582 行

坐标数组 `coords` 必须与 `gp_ids` 顺序完全一致，否则体积检查会失效。

**修改前**：
```python
coords = np.array([
    [grid_x[j, i], grid_y[j, i], bottom_z[j, i]],       # SW_bot
    [grid_x[j, i+1], grid_y[j, i+1], bottom_z[j, i+1]], # SE_bot
    [grid_x[j+1, i+1], grid_y[j+1, i+1], bottom_z[j+1, i+1]], # NE_bot ← 错误
    [grid_x[j+1, i], grid_y[j+1, i], bottom_z[j+1, i]], # NW_bot
    [grid_x[j, i], grid_y[j, i], top_z[j, i]],          # SW_top
    [grid_x[j, i+1], grid_y[j, i+1], top_z[j, i+1]],    # SE_top
    [grid_x[j+1, i+1], grid_y[j+1, i+1], top_z[j+1, i+1]], # NE_top
    [grid_x[j+1, i], grid_y[j+1, i], top_z[j+1, i]],    # NW_top
])
```

**修改后**：
```python
coords = np.array([
    [grid_x[j, i], grid_y[j, i], bottom_z[j, i]],           # 0: SW_bot (1)
    [grid_x[j, i+1], grid_y[j, i+1], bottom_z[j, i+1]],     # 1: SE_bot (2)
    [grid_x[j+1, i], grid_y[j+1, i], bottom_z[j+1, i]],     # 2: NW_bot (3) ← 修正
    [grid_x[j, i], grid_y[j, i], top_z[j, i]],              # 3: SW_top (4) ← 修正
    [grid_x[j+1, i+1], grid_y[j+1, i+1], bottom_z[j+1, i+1]], # 4: NE_bot (5) ← 修正
    [grid_x[j+1, i], grid_y[j+1, i], top_z[j+1, i]],        # 5: NW_top (6) ← 修正
    [grid_x[j, i+1], grid_y[j, i+1], top_z[j, i+1]],        # 6: SE_top (7) ← 修正
    [grid_x[j+1, i+1], grid_y[j+1, i+1], top_z[j+1, i+1]],  # 7: NE_top (8)
])
```

### 3. 更新文件头注释说明
**文件**: `src/exporters/f3grid_exporter_v2.py`
**位置**: 第 1-43 行

更新了文件头的 B8 节点顺序说明，明确标注这是**交织模式**，并提供了详细的对应关系图。

## 验证方法

### 1. 语法检查
```bash
python -m py_compile src/exporters/f3grid_exporter_v2.py
```
✅ **通过**：无语法错误

### 2. 导入测试
在修复后，重新导出模型并在 FLAC3D 中导入：
```flac3d
zone import f3grid "geological_model.f3grid"
```

**预期结果**：
- ✅ 不再出现 "Zone X has a negative volume" 错误
- ✅ 不再出现 "tet volumes <= 0" 错误
- ✅ 所有单元正确导入，无几何翻转

### 3. 测试建议
1. 使用 app_qt.py 重新导出模型
2. 在 FLAC3D 中导入新的 f3grid 文件
3. 检查导入日志，确认无负体积错误
4. 使用 `zone list information` 查看单元质量统计

## FLAC3D B8 节点顺序参考图

```
从顶视图看（Z轴向上）:

    NW(3,6) ---- NE(5,8)        y (北)
     |            |             ^
     |            |             |
    SW(1,4) ---- SE(2,7)        +---> x (东)
                               /
                              z (向上)

节点对应关系：
- 底面：1(SW), 2(SE), 3(NW), 5(NE)
- 顶面：4(SW), 7(SE), 6(NW), 8(NE)

垂直对应：
- 1(SW_bot) ↔ 4(SW_top)
- 2(SE_bot) ↔ 7(SE_top)
- 3(NW_bot) ↔ 6(NW_top)
- 5(NE_bot) ↔ 8(NE_top)
```

## 其他已实现的优化（已在代码中）

根据 `优化建议.md`，以下优化已在之前的版本中实现：

1. ✅ **角点厚度检查**（第 542-550 行）
   - 使用 `min(corner_thicknesses)` 而非平均厚度
   - 确保所有角点厚度 > MIN_ZONE_THICKNESS

2. ✅ **四面体体积检查**（第 57-102, 584-587 行）
   - 使用 5 四面体分解检查六面体几何
   - 过滤体积 <= 0 的退化单元

3. ✅ **坐标排序**（第 383-425 行）
   - 强制 X、Y 升序排列
   - 避免节点顺序反转问题

4. ✅ **统一降采样模式**（第 210-211, 228-237 行）
   - `uniform_downsample=True` 确保层间共形
   - 避免不同分辨率导致的缝隙

## 参考文档

- **FLAC3D 官方文档**: https://docs.itascacg.com
  - "Orientation of Nodes and Faces within a Zone"
  - f3grid ASCII 格式规范
- **优化建议**: `优化建议.md` (项目根目录)
- **修复计划**: `F3GRID_FIX_PLAN.md` (已过时，本次修复为最终版本)

## 结论

**核心修复**：将 B8 节点顺序从"先底面后顶面"模式改为 FLAC3D 官方的"交织模式"。

**预期效果**：
- 完全消除 "negative volume" 和 "tet volumes <= 0" 错误
- 模型几何正确，无翻转、无扭曲
- 满足 FLAC3D 标准，可直接用于计算

**下一步**：
1. 使用修复后的导出器重新导出模型
2. 在 FLAC3D 中测试导入和计算
3. 如果还有其他问题，参考 `优化建议.md` 进一步调整

---

**修复完成时间**: 2025-12-21
**版本**: f3grid_exporter_v2.py (最终修正版)
**状态**: ✅ 已验证语法，待实际测试
