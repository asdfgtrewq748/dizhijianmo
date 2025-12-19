"""
FLAC3D f3grid 导出器 v2 - 正确的网格格式

根据 FLAC3D 官方文档实现的 f3grid ASCII 格式导出器。

格式规范 (来自 https://docs.itascacg.com):
- 注释行以 * 开头
- G <id> <x> <y> <z>  - 节点定义
- Z B8 <id> <gp1> ... <gp8>  - 八节点六面体单元定义
- ZGROUP '<name>'  - 单元分组定义
  <zone_id> <zone_id> ...

B8 节点顺序 (FLAC3D 标准):

    顶面 (z+):          底面 (z-):
       7 ---- 6            3 ---- 2
      /|     /|           /|     /|
     4 ---- 5 |          0 ---- 1 |
     | 3 ---| 2          | 7 ---| 6
     |/     |/           |/     |/
     0 ---- 1            4 ---- 5

    实际 FLAC3D B8 顺序: 底面逆时针 (0,1,2,3) + 顶面逆时针 (4,5,6,7)

         5 ---- 6              y
        /|     /|              ^
       4 ---- 7 |              |
       | 1 ---| 2              +---> x
       |/     |/              /
       0 ---- 3              z (向下为负)

    索引:
    - 0: (x_min, y_min, z_min) - 底面左下
    - 1: (x_min, y_max, z_min) - 底面左上
    - 2: (x_max, y_max, z_min) - 底面右上
    - 3: (x_max, y_min, z_min) - 底面右下
    - 4: (x_min, y_min, z_max) - 顶面左下
    - 5: (x_min, y_max, z_max) - 顶面左上
    - 6: (x_max, y_max, z_max) - 顶面右上
    - 7: (x_max, y_min, z_max) - 顶面右下

关键实现要点:
1. 层间节点共享 - 上层底面 = 下层顶面，保证应力连续传递
2. 正确的 B8 节点顺序 - 按照 FLAC3D 标准
3. 单元质量检查 - 过滤退化单元
4. 坐标精度控制 - 避免浮点误差
"""

import os
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GridPoint:
    """网格节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class BrickZone:
    """B8 六面体单元"""
    id: int
    gp_ids: List[int]  # 8个节点 ID，按 FLAC3D 标准顺序
    group: str


@dataclass
class ExportStats:
    """导出统计"""
    total_gridpoints: int = 0
    total_zones: int = 0
    shared_nodes: int = 0
    degenerate_zones_removed: int = 0
    groups: int = 0
    min_thickness: float = float('inf')
    max_thickness: float = 0.0
    file_size_kb: float = 0.0


class F3GridExporterV2:
    """
    FLAC3D f3grid 导出器 v2

    正确实现层间节点共享和 B8 单元顺序。

    使用方法:
        exporter = F3GridExporterV2()
        exporter.export(data, "model.f3grid", options)

    在 FLAC3D 中导入:
        zone import f3grid "model.f3grid"
    """

    # 坐标精度（小数位数），用于节点合并判断
    COORD_PRECISION = 6

    # 最小单元厚度（米），低于此值视为退化单元
    MIN_ZONE_THICKNESS = 0.001

    def __init__(self):
        self.gridpoints: List[GridPoint] = []
        self.zones: List[BrickZone] = []
        self.groups: Dict[str, List[int]] = {}  # group_name -> [zone_ids]

        self._next_gp_id = 1
        self._next_zone_id = 1

        # 坐标 -> 节点ID 映射，用于节点共享
        self._coord_to_gp: Dict[Tuple[float, float, float], int] = {}

        self.stats = ExportStats()

    def export(self, data: Dict[str, Any], output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出地质模型为 FLAC3D f3grid 格式

        Args:
            data: 地层数据字典
                {
                    "layers": [
                        {
                            "name": str,
                            "grid_x": 1D array (nx,) 或 2D array (ny, nx),
                            "grid_y": 1D array (ny,) 或 2D array (ny, nx),
                            "top_surface_z": 2D array (ny, nx),
                            "bottom_surface_z": 2D array (ny, nx)
                        },
                        ...  # 从下到上排序
                    ]
                }
            output_path: 输出文件路径 (.f3grid)
            options: 导出选项
                - downsample_factor: int, 降采样倍数 (默认 1)
                - min_zone_thickness: float, 最小单元厚度 (默认 0.001)
                - coord_precision: int, 坐标精度 (默认 6)

        Returns:
            输出文件路径
        """
        options = options or {}
        downsample = max(1, int(options.get('downsample_factor', 1)))
        self.MIN_ZONE_THICKNESS = float(options.get('min_zone_thickness', 0.001))
        self.COORD_PRECISION = int(options.get('coord_precision', 6))

        # 重置状态
        self._reset()

        layers = data.get('layers', [])
        if not layers:
            raise ValueError("没有可导出的地层数据")

        print(f"\n{'='*60}")
        print(f"FLAC3D f3grid Exporter v2")
        print(f"{'='*60}")
        print(f"地层数量: {len(layers)}")
        print(f"降采样: {downsample}x")
        print(f"输出文件: {output_path}")

        # 1. 生成所有层的网格，实现层间节点共享
        self._generate_all_layers(layers, downsample)

        # 2. 写入 f3grid 文件
        self._write_f3grid(output_path)

        # 3. 计算文件大小
        try:
            self.stats.file_size_kb = os.path.getsize(output_path) / 1024
        except:
            pass

        # 打印统计
        self._print_stats()

        print(f"\n[OK] 导出完成!")
        print(f"\n在 FLAC3D 中使用:")
        print(f'  zone import f3grid "{os.path.basename(output_path)}"')

        return output_path

    def _reset(self):
        """重置导出器状态"""
        self.gridpoints = []
        self.zones = []
        self.groups = {}
        self._next_gp_id = 1
        self._next_zone_id = 1
        self._coord_to_gp = {}
        self.stats = ExportStats()

    def _generate_all_layers(self, layers: List[Dict], downsample: int):
        """
        生成所有层的网格，关键是实现层间节点共享

        策略:
        1. 从最底层开始，逐层向上处理
        2. 每层创建时，检查底面节点是否已存在（与下层顶面共享）
        3. 顶面节点总是新创建
        """
        print(f"\n--- 生成网格 ---")

        # 存储每层的顶面节点 ID 网格，用于下一层共享
        # last_top_gp_ids[j][i] = 节点ID
        last_top_gp_ids: Optional[List[List[int]]] = None

        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get('name', f'Layer_{layer_idx}')
            safe_name = self._sanitize_name(layer_name)

            print(f"\n  处理地层 {layer_idx+1}/{len(layers)}: {layer_name}")

            # 获取网格数据
            grid_x = np.asarray(layer['grid_x'], dtype=float)
            grid_y = np.asarray(layer['grid_y'], dtype=float)
            top_z = np.asarray(layer['top_surface_z'], dtype=float)
            bottom_z = np.asarray(layer['bottom_surface_z'], dtype=float)

            # 确保是 2D 网格
            if grid_x.ndim == 1 and grid_y.ndim == 1:
                grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # 降采样
            if downsample > 1:
                grid_x = grid_x[::downsample, ::downsample]
                grid_y = grid_y[::downsample, ::downsample]
                top_z = top_z[::downsample, ::downsample]
                bottom_z = bottom_z[::downsample, ::downsample]

            ny, nx = grid_x.shape
            print(f"    网格大小: {ny} x {nx}")

            # 创建节点网格
            # bottom_gp_ids[j][i] = 底面节点ID
            # top_gp_ids[j][i] = 顶面节点ID
            bottom_gp_ids = [[0] * nx for _ in range(ny)]
            top_gp_ids = [[0] * nx for _ in range(ny)]

            shared_count = 0
            new_bottom_count = 0

            # 创建底面节点
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(bottom_z[j, i])

                    if np.isnan(z):
                        continue

                    # 检查是否可以复用上一层的顶面节点
                    if last_top_gp_ids is not None and j < len(last_top_gp_ids) and i < len(last_top_gp_ids[j]):
                        existing_gp_id = last_top_gp_ids[j][i]
                        if existing_gp_id > 0:
                            # 找到上一层的顶面节点
                            existing_gp = self._get_gridpoint(existing_gp_id)
                            if existing_gp is not None:
                                # 检查坐标是否匹配
                                if (abs(existing_gp.x - x) < 1e-6 and
                                    abs(existing_gp.y - y) < 1e-6 and
                                    abs(existing_gp.z - z) < 1e-6):
                                    # 完全匹配，复用节点
                                    bottom_gp_ids[j][i] = existing_gp_id
                                    shared_count += 1
                                    continue

                    # 需要创建新节点
                    gp_id = self._get_or_create_gridpoint(x, y, z)
                    bottom_gp_ids[j][i] = gp_id
                    new_bottom_count += 1

            # 创建顶面节点（总是新创建）
            for j in range(ny):
                for i in range(nx):
                    x = float(grid_x[j, i])
                    y = float(grid_y[j, i])
                    z = float(top_z[j, i])

                    if np.isnan(z):
                        continue

                    gp_id = self._get_or_create_gridpoint(x, y, z)
                    top_gp_ids[j][i] = gp_id

            print(f"    底面节点: {new_bottom_count} 新建, {shared_count} 共享")
            self.stats.shared_nodes += shared_count

            # 创建单元
            zone_count = 0
            degenerate_count = 0

            if safe_name not in self.groups:
                self.groups[safe_name] = []

            for j in range(ny - 1):
                for i in range(nx - 1):
                    # 获取 8 个角点的节点 ID
                    # 底面 4 个角点
                    gp_b_00 = bottom_gp_ids[j][i]         # (i, j) - 左下
                    gp_b_10 = bottom_gp_ids[j][i+1]       # (i+1, j) - 右下
                    gp_b_01 = bottom_gp_ids[j+1][i]       # (i, j+1) - 左上
                    gp_b_11 = bottom_gp_ids[j+1][i+1]     # (i+1, j+1) - 右上

                    # 顶面 4 个角点
                    gp_t_00 = top_gp_ids[j][i]
                    gp_t_10 = top_gp_ids[j][i+1]
                    gp_t_01 = top_gp_ids[j+1][i]
                    gp_t_11 = top_gp_ids[j+1][i+1]

                    # 检查所有节点是否有效
                    gp_all = [gp_b_00, gp_b_10, gp_b_01, gp_b_11,
                              gp_t_00, gp_t_10, gp_t_01, gp_t_11]

                    if any(gp == 0 for gp in gp_all):
                        continue  # 有无效节点，跳过

                    # 检查是否为退化单元（所有节点不能重复）
                    if len(set(gp_all)) < 8:
                        degenerate_count += 1
                        continue

                    # 检查单元厚度
                    z_bottom = [bottom_z[j, i], bottom_z[j, i+1],
                                bottom_z[j+1, i], bottom_z[j+1, i+1]]
                    z_top = [top_z[j, i], top_z[j, i+1],
                             top_z[j+1, i], top_z[j+1, i+1]]

                    if any(np.isnan(z) for z in z_bottom + z_top):
                        continue

                    avg_thickness = sum(z_top) / 4 - sum(z_bottom) / 4

                    if avg_thickness < self.MIN_ZONE_THICKNESS:
                        degenerate_count += 1
                        continue

                    # 更新厚度统计
                    self.stats.min_thickness = min(self.stats.min_thickness, avg_thickness)
                    self.stats.max_thickness = max(self.stats.max_thickness, avg_thickness)

                    # 创建 B8 单元
                    # FLAC3D B8 节点顺序:
                    #   底面: 0(左下), 1(左上), 2(右上), 3(右下) - 逆时针
                    #   顶面: 4(左下), 5(左上), 6(右上), 7(右下) - 逆时针
                    gp_ids = [
                        gp_b_00,  # 0: 底面左下 (x_min, y_min, z_min)
                        gp_b_01,  # 1: 底面左上 (x_min, y_max, z_min)
                        gp_b_11,  # 2: 底面右上 (x_max, y_max, z_min)
                        gp_b_10,  # 3: 底面右下 (x_max, y_min, z_min)
                        gp_t_00,  # 4: 顶面左下 (x_min, y_min, z_max)
                        gp_t_01,  # 5: 顶面左上 (x_min, y_max, z_max)
                        gp_t_11,  # 6: 顶面右上 (x_max, y_max, z_max)
                        gp_t_10,  # 7: 顶面右下 (x_max, y_min, z_max)
                    ]

                    zone = BrickZone(
                        id=self._next_zone_id,
                        gp_ids=gp_ids,
                        group=safe_name
                    )
                    self.zones.append(zone)
                    self.groups[safe_name].append(zone.id)
                    self._next_zone_id += 1
                    zone_count += 1

            print(f"    单元数: {zone_count}")
            if degenerate_count > 0:
                print(f"    过滤退化单元: {degenerate_count}")
                self.stats.degenerate_zones_removed += degenerate_count

            # 保存顶面节点 ID，供下一层使用
            last_top_gp_ids = top_gp_ids

        # 更新统计
        self.stats.total_gridpoints = len(self.gridpoints)
        self.stats.total_zones = len(self.zones)
        self.stats.groups = len(self.groups)

    def _get_or_create_gridpoint(self, x: float, y: float, z: float) -> int:
        """获取或创建节点，自动处理坐标合并"""
        # 四舍五入到指定精度
        key = (
            round(x, self.COORD_PRECISION),
            round(y, self.COORD_PRECISION),
            round(z, self.COORD_PRECISION)
        )

        if key in self._coord_to_gp:
            return self._coord_to_gp[key]

        gp = GridPoint(
            id=self._next_gp_id,
            x=key[0],
            y=key[1],
            z=key[2]
        )
        self.gridpoints.append(gp)
        self._coord_to_gp[key] = gp.id
        self._next_gp_id += 1

        return gp.id

    def _get_gridpoint(self, gp_id: int) -> Optional[GridPoint]:
        """根据 ID 获取节点"""
        for gp in self.gridpoints:
            if gp.id == gp_id:
                return gp
        return None

    def _write_f3grid(self, output_path: str):
        """写入 f3grid ASCII 文件"""
        print(f"\n--- 写入文件 ---")

        with open(output_path, 'w', encoding='utf-8') as f:
            # 文件头
            f.write("* ============================================================\n")
            f.write("* FLAC3D Native Grid File (f3grid ASCII format)\n")
            f.write("* Generated by F3GridExporterV2\n")
            f.write("* ============================================================\n")
            f.write(f"* Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"* Total Gridpoints: {len(self.gridpoints)}\n")
            f.write(f"* Total Zones: {len(self.zones)}\n")
            f.write(f"* Total Groups: {len(self.groups)}\n")
            f.write("* ============================================================\n")
            f.write("*\n")
            f.write("* Usage in FLAC3D:\n")
            f.write(f'*   zone import f3grid "{os.path.basename(output_path)}"\n')
            f.write("*\n")
            f.write("* ============================================================\n\n")

            # 节点定义
            f.write("* GRIDPOINTS\n")
            f.write("* G <id> <x> <y> <z>\n")
            for gp in self.gridpoints:
                f.write(f"G {gp.id} {gp.x:.6f} {gp.y:.6f} {gp.z:.6f}\n")
            f.write("\n")

            # 单元定义
            f.write("* ZONES (B8 = 8-node brick)\n")
            f.write("* Z B8 <id> <gp1> <gp2> <gp3> <gp4> <gp5> <gp6> <gp7> <gp8>\n")
            for zone in self.zones:
                gp_str = ' '.join(str(gp_id) for gp_id in zone.gp_ids)
                f.write(f"Z B8 {zone.id} {gp_str}\n")
            f.write("\n")

            # 分组定义
            f.write("* ZONE GROUPS\n")
            f.write("* ZGROUP '<name>'\n")
            f.write("* <zone_id> <zone_id> ...\n")
            for group_name, zone_ids in self.groups.items():
                if not zone_ids:
                    continue
                f.write(f"ZGROUP '{group_name}'\n")
                # 每行最多 15 个 ID
                for i in range(0, len(zone_ids), 15):
                    batch = zone_ids[i:i+15]
                    f.write(' '.join(str(zid) for zid in batch) + '\n')
                f.write("\n")

            # 文件尾
            f.write("* ============================================================\n")
            f.write("* End of Grid File\n")
            f.write("* ============================================================\n")

        print(f"  写入完成: {output_path}")

    def _sanitize_name(self, name: str) -> str:
        """将名称转换为 FLAC3D 兼容的 ASCII 形式，同岩性合并到同一组"""
        # 中文到英文的映射
        replacements = {
            '煤': 'coal',
            '砂质泥岩': 'sandy_mudstone',
            '炭质泥岩': 'carbonaceous_mudstone',
            '高岭质泥岩': 'kaolinite_mudstone',
            '高岭岩': 'kaolinite_rock',
            '风化煤': 'weathered_coal',
            '含砾': 'conglomeratic',
            '泥岩': 'mudstone',
            '砂岩': 'sandstone',
            '灰岩': 'limestone',
            '石灰岩': 'limestone',
            '页岩': 'shale',
            '粉砂岩': 'siltstone',
        }

        result = name or 'group'

        # 先去掉名称中的数字序号（如 "砂岩_25" -> "砂岩", "16-4煤" -> "煤"）
        # 去掉开头的数字和分隔符（如 "16-4煤" -> "煤", "16_3_煤" -> "煤"）
        result = re.sub(r'^[\d_\-\.]+', '', result)
        # 去掉末尾的数字和分隔符（如 "砂岩_25" -> "砂岩", "泥岩-3" -> "泥岩"）
        result = re.sub(r'[\d_\-\.]+$', '', result)

        # 中文转英文
        for cn, en in replacements.items():
            result = result.replace(cn, en)

        # 替换非法字符
        result = re.sub(r'[^0-9A-Za-z_]', '_', result)
        result = result.strip('_')

        # 再次清理可能残留的数字后缀（英文转换后可能还有，如 "sandstone_25"）
        result = re.sub(r'_*\d+$', '', result)
        result = result.strip('_')

        return result if result else 'group'

    def _print_stats(self):
        """打印导出统计"""
        print(f"\n{'='*60}")
        print(f"导出统计")
        print(f"{'='*60}")
        print(f"总节点数: {self.stats.total_gridpoints:,}")
        print(f"总单元数: {self.stats.total_zones:,}")
        print(f"共享节点数: {self.stats.shared_nodes:,}")
        print(f"过滤退化单元: {self.stats.degenerate_zones_removed:,}")
        print(f"分组数: {self.stats.groups}")

        if self.stats.min_thickness < float('inf'):
            print(f"厚度范围: {self.stats.min_thickness:.3f}m - {self.stats.max_thickness:.3f}m")

        print(f"\n分组详情:")
        for name, zone_ids in self.groups.items():
            print(f"  {name}: {len(zone_ids):,} 单元")


def export_f3grid(data: Dict[str, Any], output_path: str,
                  options: Optional[Dict[str, Any]] = None) -> str:
    """便捷导出函数"""
    exporter = F3GridExporterV2()
    return exporter.export(data, output_path, options)


# 测试代码
if __name__ == '__main__':
    import numpy as np

    # 创建测试数据
    nx, ny = 10, 10
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    XI, YI = np.meshgrid(x, y)

    # 创建两个层
    base_z = -100 + 0.1 * XI + 0.05 * YI

    layers = [
        {
            'name': '底板泥岩',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 5,
            'bottom_surface_z': base_z,
        },
        {
            'name': '煤层',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 8,
            'bottom_surface_z': base_z + 5,  # 与底板泥岩顶面一致
        },
        {
            'name': '顶板砂岩',
            'grid_x': x,
            'grid_y': y,
            'top_surface_z': base_z + 20,
            'bottom_surface_z': base_z + 8,  # 与煤层顶面一致
        },
    ]

    data = {'layers': layers}

    exporter = F3GridExporterV2()
    exporter.export(data, 'test_model.f3grid')
