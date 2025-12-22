# FPN 格式导出使用指南

## 什么是 FPN 格式？

FPN (Finite element Program Neutral file) 是 Midas GTS NX 的标准网格交换格式，是一种中间格式，可以被多种转换工具读取和转换。

## 为什么使用 FPN 格式？

1. **更通用**：FPN 是行业标准格式，兼容性好
2. **可转换**：可以使用专业转换工具转换为 FLAC3D f3grid 格式
3. **更简单**：节点顺序问题更少，格式更清晰
4. **更稳定**：避免直接导出 f3grid 可能遇到的节点顺序问题

## 使用方法

### 1. 在应用中导出

1. 打开应用 `python app_qt.py`
2. 完成地质建模
3. 选择 **导出 FLAC3D**
4. 在格式下拉菜单中选择 **FPN (中间格式)**
5. 设置降采样倍数（推荐 2-3x）
6. 点击导出，保存为 `.fpn` 文件

### 2. 转换为 FLAC3D f3grid

#### 方法1：使用 Midas GTS NX（推荐）

```
1. 打开 Midas GTS NX
2. File -> Import -> Neutral File (.fpn)
3. 选择导出的 FPN 文件
4. File -> Export -> FLAC3D (.f3grid)
5. 导出为 f3grid 格式
```

#### 方法2：使用其他转换工具

- **Gmsh**: 开源网格处理工具
- **Ansys**: 可以导入 FPN 并导出多种格式
- **Abaqus**: 支持 FPN 导入

### 3. 在 FLAC3D 中使用

转换后的 f3grid 文件可以在 FLAC3D 中直接导入：

```flac
zone import f3grid "converted_model.f3grid"
zone list information
zone plot
```

## FPN 文件格式说明

FPN 是纯文本格式，结构如下：

```
$$ Neutral File Header
VER, 2.0.0

$$ Unit system
UNIT, KN,M,SEC

$$ Node definitions
NODE   , 1, 0.0, 0.0, 0.0, 1, , ,
NODE   , 2, 74.16, 0.0, 0.0, 1, , ,
...

$$ Element definitions (8-node hexahedron)
HEXA   , 1, 1, 1, 2, 52, 51, 2501, 2502
       , 2552, 2551, , , , , ,
```

## 节点顺序

FPN 六面体单元使用标准的节点顺序：

```
底面（逆时针，从下向上看）:
  NW(3) ---- NE(2)
   |          |
  SW(0) ---- SE(1)

顶面（逆时针）:
  NW(7) ---- NE(6)
   |          |
  SW(4) ---- SE(5)

节点顺序: [SW, SE, NE, NW, SW, SE, NE, NW]
```

## 优势对比

| 特性 | FPN 格式 | 直接 f3grid |
|------|----------|-------------|
| 节点顺序标准化 | ✓ 标准 | ✗ 可能不兼容 |
| 转换灵活性 | ✓ 多种工具 | ✗ 无转换选项 |
| 错误诊断 | ✓ 易于检查 | ✗ 难以调试 |
| FLAC3D版本兼容 | ✓ 通用 | ✗ 版本相关 |
| 几何验证 | ✓ 转换时验证 | ✗ 导入时才发现 |

## 常见问题

### Q: 转换后单元还是有负体积？

A: 检查转换工具的节点顺序设置，确保使用"逆时针"或"右手法则"。

### Q: 文件太大怎么办？

A:
1. 增加降采样倍数（2x, 3x, 5x）
2. 只选择关键煤层使用高密度网格
3. 使用分层导出

### Q: 转换工具在哪下载？

A:
- Gmsh: https://gmsh.info/ (免费开源)
- Midas GTS NX: 需要商业许可证

## 技术支持

如有问题，请提供：
1. FPN 文件样本（前100行）
2. FLAC3D 错误信息
3. 使用的转换工具和版本

---

**创建日期**: 2025-12-20
**版本**: 1.0
**作者**: Claude Code
