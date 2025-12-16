"""
FLAC3D Native Grid Exporter (.f3grid)

直接导出FLAC3D原生网格格式,彻底避免STL几何冲突问题

核心优势:
1. 拓扑直接定义 - 无需geometry import,直接定义节点和单元
2. 层间节点共享 - 上层底面节点ID = 下层顶面节点ID,保证应力传递
3. 无几何冲突 - 不依赖三角面片,避免FLAC3D网格生成时的体积冲突
4. 文本格式 - 便于调试和验证

格式详见: docs/F3GRID_FORMAT_SPEC.md
"""

import os
import re
from builtins import ValueError
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from .base_exporter import BaseExporter


@dataclass
class GridPoint:
    """网格节点"""
    id: int
    x: float
    y: float
    z: float


@dataclass
class Zone:
    """网格单元(BRICK六面体)"""
    id: int
    type: str  # "brick"
    gridpoint_ids: List[int]  # 8个节点ID,顺序: bottom(0-1-2-3逆时针) + top(4-5-6-7逆时针)
    group: str  # 所属layer名称


@dataclass
class ZoneGroup:
    """单元分组"""
    name: str
    zone_ids: List[int]


class F3GridExporter(BaseExporter):
    """
    FLAC3D原生网格格式(.f3grid)导出器
    
    工作流程:
    1. 对每一层调用_generate_layer_grid()生成独立网格
    2. 调用_merge_layers()合并所有层,复用层间节点
    3. 调用_write_f3grid()写入文本格式文件
    
    层间节点共享策略:
    - 上层底面节点 = 下层顶面节点 (同一(x,y)位置)
    - 通过节点ID复用实现拓扑连续
    - 确保Z坐标完全一致(已由enforce_columnwise_order保证)
    """
    
    def __init__(self):
        super().__init__()
        self.gridpoints: List[GridPoint] = []
        self.zones: List[Zone] = []
        self.groups: List[ZoneGroup] = []
        self._next_gp_id = 1  # 节点ID计数器
        self._next_zone_id = 1  # 单元ID计数器
        self._gridpoint_lookup: Dict[int, GridPoint] = {}
        self.interface_tolerance = 1e-4
    
    def export(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出多层地质模型为FLAC3D原生网格格式
        
        Args:
            data: 包含地层数据的字典,格式:
                {
                    "layers": [
                        {
                            "name": "LayerName",
                            "grid_x": np.ndarray (shape: [ny, nx]),
                            "grid_y": np.ndarray (shape: [ny, nx]),
                            "top_surface_z": np.ndarray (shape: [ny, nx]),
                            "bottom_surface_z": np.ndarray (shape: [ny, nx])
                        },
                        ...
                    ]
                }
            output_path: 输出文件路径(.f3grid)
            options: 导出选项
                - downsample_factor: 降采样倍数(默认5,减少单元数量)
                - coordinate_offset: 坐标偏移量(默认None,用于大坐标归一化)
        
        Returns:
            str: 输出文件路径
        """
        # 解析选项
        options = options or {}
        downsample = int(options.get('downsample_factor') or options.get('downsample') or 1)
        raw_offset = options.get('coordinate_offset')
        if raw_offset is None:
            raw_offset = options.get('coord_offset')
        normalize_coords = bool(options.get('normalize_coords'))
        self.interface_tolerance = float(options.get('interface_tolerance', 1e-4))
        downsample = max(1, downsample)
        filter_bad_zones = bool(options.get('filter_bad_zones') or options.get('enforce_zone_quality'))
        min_zone_thickness = float(options.get('min_zone_thickness', 1e-3))

        # 获取layers
        layers = data.get('layers', [])
        if not layers:
            raise ValueError("No layers found in data")

        # 可选：在最上层外添加一个平顶封顶层，避免.f3grid导入时没有封顶面
        add_top_cap = bool(options.get('add_top_cap') or options.get('top_cap', False))
        top_cap_thickness = float(options.get('top_cap_thickness', 1.0))
        top_cap_z = options.get('top_cap_z', None)  # 如果提供绝对高度则优先使用
        top_cap_name = options.get('top_cap_name', 'TopCap')
        if add_top_cap:
            # 假定输入layers按从下到上排序
            top_layer = layers[-1]
            gx = np.asarray(top_layer['grid_x'], dtype=float)
            gy = np.asarray(top_layer['grid_y'], dtype=float)
            top_surface = np.asarray(top_layer['top_surface_z'], dtype=float)

            # 计算封顶层的顶面高度：如果提供top_cap_z则使用该常数，否则使用top_surface + thickness
            if top_cap_z is not None:
                try:
                    cap_top_value = float(top_cap_z)
                    cap_top = np.full_like(top_surface, cap_top_value, dtype=float)
                except Exception:
                    cap_top = top_surface + top_cap_thickness
            else:
                cap_top = top_surface + top_cap_thickness

            cap_bottom = top_surface.copy()

            cap_layer = {
                'name': top_cap_name,
                'grid_x': gx,
                'grid_y': gy,
                'top_surface_z': cap_top,
                'bottom_surface_z': cap_bottom,
            }
            layers = list(layers) + [cap_layer]
            print(f"[F3GRID Export] Added top cap layer '{top_cap_name}' thickness ~{top_cap_thickness} m")

        if raw_offset is not None:
            if len(raw_offset) != 3:
                raise ValueError("coordinate_offset 必须是长度为3的(x,y,z)序列")
            coord_offset = tuple(float(v) for v in raw_offset)
            print(f"[F3GRID Export] 使用外部坐标偏移: {coord_offset}")
        elif normalize_coords:
            coord_offset = self._compute_auto_offset(layers)
            print(f"[F3GRID Export] normalize_coords=True, 偏移量: {coord_offset}")
        else:
            coord_offset = (0.0, 0.0, 0.0)
        
        # 重置数据
        self.gridpoints = []
        self.zones = []
        self.groups = []
        self._next_gp_id = 1
        self._next_zone_id = 1
        self._gridpoint_lookup = {}
        
        # 验证数据
        for i, layer in enumerate(layers):
            required_keys = ['name', 'grid_x', 'grid_y', 'top_surface_z', 'bottom_surface_z']
            for key in required_keys:
                if key not in layer:
                    raise ValueError(f"Layer {i} missing required key: {key}")
        
        print(f"\n=== F3GRID Export Started ===")
        print(f"Layers: {len(layers)}")
        print(f"Downsample factor: {downsample}")
        print(f"Output: {output_path}")
        
        # Step 3 - 生成每层的网格
        layer_grids = []
        for layer in layers:
            grid = self._generate_layer_grid(layer, downsample, coord_offset)
            layer_grids.append(grid)
        
        # Step 4 - 合并所有层(实现层间节点共享)
        self._merge_layers(layer_grids)
        
        if filter_bad_zones:
            self._filter_degenerate_zones(min_zone_thickness)

        # Step 5 - 写入.f3grid文件
        self._write_f3grid(output_path)
        
        print(f"Total GridPoints: {len(self.gridpoints)}")
        print(f"Total Zones: {len(self.zones)}")
        print(f"Total Groups: {len(self.groups)}")
        print(f"=== F3GRID Export Completed ===\n")
        print(f" 👉 在 FLAC3D 中使用: zone import f3grid \"{output_path}\"")
        
        return output_path
    
    def _generate_layer_grid(self, layer: Dict[str, Any], downsample: int, coord_offset: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        为单层生成网格数据(节点+单元)
        
        Args:
            layer: 单层数据,包含grid_x, grid_y, top_surface_z, bottom_surface_z
            downsample: 降采样倍数
            coord_offset: 坐标偏移量(x_offset, y_offset, z_offset)
        
        Returns:
            Dict: 包含该层的节点和单元数据
                {
                    "name": str,
                    "top_nodes": List[GridPoint],  # 顶面节点 (shape: [ny, nx])
                    "bottom_nodes": List[GridPoint],  # 底面节点 (shape: [ny, nx])
                    "zones": List[Zone],  # 单元列表
                    "nx": int,  # X方向节点数
                    "ny": int   # Y方向节点数
                }
        """
        layer_name = layer['name']
        
        # 1. 降采样
        grid_x = self._downsample_grid(layer['grid_x'], downsample)
        grid_y = self._downsample_grid(layer['grid_y'], downsample)
        top_z = self._downsample_grid(layer['top_surface_z'], downsample)
        bottom_z = self._downsample_grid(layer['bottom_surface_z'], downsample)
        
        ny, nx = grid_x.shape
        
        # 2. 应用坐标偏移
        if coord_offset is None:
            coord_offset = (0.0, 0.0, 0.0)
        x_off, y_off, z_off = coord_offset
        
        # 3. 创建底面节点 (二维数组: [ny, nx])
        bottom_nodes = []
        for j in range(ny):
            row = []
            for i in range(nx):
                gp = GridPoint(
                    id=self._next_gp_id,
                    x=grid_x[j, i] - x_off,
                    y=grid_y[j, i] - y_off,
                    z=bottom_z[j, i] - z_off
                )
                self._next_gp_id += 1
                row.append(gp)
            bottom_nodes.append(row)
        
        # 4. 创建顶面节点 (二维数组: [ny, nx])
        top_nodes = []
        for j in range(ny):
            row = []
            for i in range(nx):
                gp = GridPoint(
                    id=self._next_gp_id,
                    x=grid_x[j, i] - x_off,
                    y=grid_y[j, i] - y_off,
                    z=top_z[j, i] - z_off
                )
                self._next_gp_id += 1
                row.append(gp)
            top_nodes.append(row)
        
        # 5. 创建BRICK单元
        # 对每个格子 (j, i)，创建一个BRICK
        # 底面4个角: (j,i), (j,i+1), (j+1,i+1), (j+1,i)
        # 节点编号规则(逆时针):
        #   bottom: [sw, se, ne, nw] = [(j,i), (j,i+1), (j+1,i+1), (j+1,i)]
        #   top:    [sw, se, ne, nw] = 同样位置
        zones = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                # 底面4个节点 (逆时针: sw->se->ne->nw)
                gp_bottom = [
                    bottom_nodes[j][i].id,      # sw (j, i)
                    bottom_nodes[j][i+1].id,    # se (j, i+1)
                    bottom_nodes[j+1][i+1].id,  # ne (j+1, i+1)
                    bottom_nodes[j+1][i].id     # nw (j+1, i)
                ]
                
                # 顶面4个节点 (逆时针: sw->se->ne->nw)
                gp_top = [
                    top_nodes[j][i].id,      # sw
                    top_nodes[j][i+1].id,    # se
                    top_nodes[j+1][i+1].id,  # ne
                    top_nodes[j+1][i].id     # nw
                ]
                
                zone = self._create_brick_zone(gp_bottom, gp_top, layer_name)
                zones.append(zone)
        
        print(f"  Layer '{layer_name}': {nx}x{ny} nodes, {len(zones)} zones")
        
        return {
            "name": layer_name,
            "top_nodes": top_nodes,      # [ny][nx]
            "bottom_nodes": bottom_nodes, # [ny][nx]
            "zones": zones,
            "nx": nx,
            "ny": ny
        }

    def _compute_auto_offset(self, layers: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """根据所有层数据估算一个稳定的坐标偏移(中位数/最小值)"""
        x_values: List[float] = []
        y_values: List[float] = []
        z_values: List[float] = []

        for layer in layers:
            for field, collector, reducer in [
                ('grid_x', x_values, 'median'),
                ('grid_y', y_values, 'median'),
                ('top_surface_z', z_values, 'min'),
                ('bottom_surface_z', z_values, 'min')
            ]:
                arr = layer.get(field)
                if arr is None:
                    continue
                data = np.asarray(arr, dtype=float)
                valid = data[np.isfinite(data)]
                if valid.size == 0:
                    continue
                if reducer == 'median':
                    collector.append(float(np.median(valid)))
                else:
                    collector.append(float(np.min(valid)))

        if x_values and y_values and z_values:
            return (
                float(np.median(x_values)),
                float(np.median(y_values)),
                float(np.min(z_values))
            )
        return (0.0, 0.0, 0.0)
    
    def _merge_layers(self, layer_grids: List[Dict[str, Any]]) -> None:
        """
        合并多层网格,实现层间节点共享
        
        核心策略:
        1. 从下往上遍历各层
        2. 对于第i层的top_nodes和第i+1层的bottom_nodes:
           - 检查(x,y)坐标是否匹配
           - 检查Z坐标是否一致(应该已由enforce_columnwise_order保证)
           - 复用节点ID(上层底面使用下层顶面的节点ID)
        3. 更新上层的zone的gridpoint_ids(0-3节点用下层的top节点ID)
        
        Args:
            layer_grids: 各层的网格数据列表(从下到上排序)
        """
        if not layer_grids:
            return
        
        print(f"\n--- Merging {len(layer_grids)} layers ---")
        
        # 1. 验证各层网格尺寸一致
        nx0, ny0 = layer_grids[0]['nx'], layer_grids[0]['ny']
        for i, grid in enumerate(layer_grids):
            if grid['nx'] != nx0 or grid['ny'] != ny0:
                raise ValueError(
                    f"Layer {i} grid size mismatch: "
                    f"expected ({nx0}, {ny0}), got ({grid['nx']}, {grid['ny']})"
                )
        
        # 2. 处理第一层(最下层): 添加所有节点和单元
        bottom_grid = layer_grids[0]
        
        # 添加底面节点
        for row in bottom_grid['bottom_nodes']:
            for gp in row:
                self.gridpoints.append(gp)
                self._gridpoint_lookup[gp.id] = gp
        
        # 添加顶面节点
        for row in bottom_grid['top_nodes']:
            for gp in row:
                self.gridpoints.append(gp)
                self._gridpoint_lookup[gp.id] = gp
        
        # 添加单元
        self.zones.extend(bottom_grid['zones'])
        
        # 创建group
        self.groups.append(ZoneGroup(
            name=bottom_grid['name'],
            zone_ids=[z.id for z in bottom_grid['zones']]
        ))
        
        print(f"  Layer 0 '{bottom_grid['name']}': added {len(bottom_grid['zones'])} zones")
        
        # 3. 处理后续各层: 复用interface节点
        for layer_idx in range(1, len(layer_grids)):
            lower_grid = layer_grids[layer_idx - 1]
            upper_grid = layer_grids[layer_idx]
            
            print(f"\n  Processing layer {layer_idx} '{upper_grid['name']}'...")
            
            # 3.1 创建(x,y) -> 下层top节点ID的映射
            # 用于快速查找interface节点
            lower_top_map = {}  # (x, y) -> GridPoint.id
            for j in range(ny0):
                for i in range(nx0):
                    gp = lower_grid['top_nodes'][j][i]
                    key = (round(gp.x, 6), round(gp.y, 6))  # 坐标取整到微米级
                    lower_top_map[key] = gp.id
            
            # 3.2 创建上层bottom节点的ID映射: old_id -> new_id(复用下层top)
            bottom_id_remap = {}  # old_bottom_id -> reused_top_id
            z_diff_max = 0.0
            z_diff_count = 0
            
            for j in range(ny0):
                for i in range(nx0):
                    old_gp = upper_grid['bottom_nodes'][j][i]
                    key = (round(old_gp.x, 6), round(old_gp.y, 6))
                    
                    if key not in lower_top_map:
                        raise ValueError(
                            f"Layer {layer_idx} bottom node at ({old_gp.x}, {old_gp.y}) "
                            f"has no matching lower layer top node"
                        )
                    
                    # 复用下层顶面节点ID
                    reused_id = lower_top_map[key]
                    bottom_id_remap[old_gp.id] = reused_id
                    
                    # 验证Z坐标一致性
                    lower_top_gp = self._gridpoint_lookup[reused_id]
                    z_diff = abs(old_gp.z - lower_top_gp.z)
                    z_diff_max = max(z_diff_max, z_diff)
                    z_diff_count += 1
                    if z_diff > self.interface_tolerance:
                        raise ValueError(
                            f"层间节点不连续: ({old_gp.x}, {old_gp.y}) diff={z_diff:.6f}m "
                            f"> tol {self.interface_tolerance:.6f}m"
                        )
            
            print(f"    Interface nodes: {z_diff_count} matched, max Z diff: {z_diff_max:.6f}m")
            
            # 3.3 添加上层的顶面节点(新节点)
            for row in upper_grid['top_nodes']:
                for gp in row:
                    self.gridpoints.append(gp)
                    self._gridpoint_lookup[gp.id] = gp
            
            # 3.4 更新上层zones的gridpoint_ids
            # 底面4个节点(0-3)使用复用ID,顶面4个节点(4-7)保持原ID
            for zone in upper_grid['zones']:
                new_gp_ids = []
                for idx, old_id in enumerate(zone.gridpoint_ids):
                    if idx < 4:  # 底面节点
                        new_gp_ids.append(bottom_id_remap[old_id])
                    else:  # 顶面节点
                        new_gp_ids.append(old_id)
                zone.gridpoint_ids = new_gp_ids
            
            # 3.5 添加上层zones和group
            self.zones.extend(upper_grid['zones'])
            self.groups.append(ZoneGroup(
                name=upper_grid['name'],
                zone_ids=[z.id for z in upper_grid['zones']]
            ))
            
            print(f"    Added {len(upper_grid['zones'])} zones")
        
        print(f"\n--- Merge completed: {len(self.gridpoints)} nodes, {len(self.zones)} zones, {len(self.groups)} groups ---")

    def _filter_degenerate_zones(self, min_thickness: float) -> None:
        """剔除重复节点或厚度过薄的单元, 缓解FLAC3D几何警告。"""
        if not self.zones:
            return

        filtered: List[Zone] = []
        removed = 0
        for zone in self.zones:
            gp_ids = zone.gridpoint_ids
            if len(set(gp_ids)) < len(gp_ids):
                removed += 1
                continue

            bottom_z = [self._gridpoint_lookup[gid].z for gid in gp_ids[:4]]
            top_z = [self._gridpoint_lookup[gid].z for gid in gp_ids[4:]]
            thickness = (sum(top_z) / 4.0) - (sum(bottom_z) / 4.0)
            if thickness <= min_thickness:
                removed += 1
                continue
            filtered.append(zone)

        if removed:
            print(
                f"[F3GRID Export] Removed {removed} degenerate zones (min thickness {min_thickness} m)"
            )
            valid_zone_ids = {zone.id for zone in filtered}
            for group in self.groups:
                group.zone_ids = [zid for zid in group.zone_ids if zid in valid_zone_ids]

        self.zones = filtered
    
    def _write_f3grid(self, output_path: str) -> None:
        """
        写入FLAC3D网格文件(.f3grid文本格式)
        
        文件结构:
        ; FLAC3D Grid File
        ; Generated by CoalSeam3D System
        ; ...metadata...
        
        GRIDPOINTS
        <id> <x> <y> <z>
        ...
        
        ZONES brick
        <id> <gp0> <gp1> <gp2> <gp3> <gp4> <gp5> <gp6> <gp7>
        ...
        
        GROUPS
        <group_name>
        <zone_id> <zone_id> ...
        ...
        
        Args:
            output_path: 输出文件路径
        """
        from datetime import datetime
        
        def _sanitize_group_name(name: str) -> str:
            """将组名转换为FLAC3D易解析的ASCII形式。"""
            replacements = {
                "煤": "coal",
                "砂质泥岩": "sandy_mudstone",
                "炭质泥岩": "carbonaceous_mudstone",
                "高岭质泥岩": "kaolinite_mudstone",
                "高岭岩": "kaolinite_rock",
                "风化煤": "weathered_coal",
                "含砾": "conglomeratic",
                "泥岩": "mudstone",
                "砂岩": "sandstone"
            }
            sanitized = name or "group"
            for key, value in replacements.items():
                sanitized = sanitized.replace(key, value)
            sanitized = re.sub(r"[^0-9A-Za-z_ ]", "_", sanitized)
            sanitized = sanitized.strip() or "group"
            return sanitized

        with open(output_path, 'w', encoding='utf-8') as f:
            # 1. 文件头和元数据(符合FLAC3D注释约定)
            f.write("* ====================================\n")
            f.write("* FLAC3D Native Grid File\n")
            f.write("* Generated by CoalSeam3D System\n")
            f.write("* ====================================\n")
            f.write(f"* Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"* Total GridPoints: {len(self.gridpoints)}\n")
            f.write(f"* Total Zones: {len(self.zones)}\n")
            f.write(f"* Total Groups: {len(self.groups)}\n")
            f.write("* ====================================\n\n")

            # 2. GRIDPOINTS: 以G开头
            f.write("* GRIDPOINTS\n")
            f.write("*   G <id> <x> <y> <z>\n")
            for gp in self.gridpoints:
                f.write(f"G {gp.id:d} {gp.x:.6f} {gp.y:.6f} {gp.z:.6f}\n")
            f.write("\n")

            # 3. ZONES: 以Z B8开头
            f.write("* ZONES (brick)\n")
            f.write("*   Z B8 <id> <gp0> <gp1> <gp2> <gp3> <gp4> <gp5> <gp6> <gp7>\n")
            for zone in self.zones:
                gp_ids = ' '.join(str(gid) for gid in zone.gridpoint_ids)
                f.write(f"Z B8 {zone.id:d} {gp_ids}\n")
            f.write("\n")

            # 4. GROUPS: 使用ZGROUP
            if self.groups:
                f.write("* ZONE GROUPS\n")
                f.write("*   ZGROUP 'name'\n")
                f.write("*   <zone_id> <zone_id> ...\n")
                for group in self.groups:
                    safe_name = _sanitize_group_name(group.name)
                    f.write(f"ZGROUP '{safe_name}'\n")
                    zone_ids = sorted(group.zone_ids)
                    max_ids = 15
                    line_ids: List[str] = []
                    for zid in zone_ids:
                        line_ids.append(str(zid))
                        if len(line_ids) >= max_ids:
                            f.write(' '.join(line_ids) + "\n")
                            line_ids = []
                    if line_ids:
                        f.write(' '.join(line_ids) + "\n")
                    f.write("\n")

            # 5. 文件尾注释
            f.write("* ====================================\n")
            f.write("* End of Grid File\n")
            f.write("* ====================================\n")
        
        print(f"  Written to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    def _downsample_grid(self, grid: np.ndarray, factor: int) -> np.ndarray:
        """
        降采样网格数据
        
        Args:
            grid: 原始网格(2D数组)
            factor: 降采样倍数(每factor个点取1个)
        
        Returns:
            降采样后的网格
        """
        if factor <= 1:
            return grid
        return grid[::factor, ::factor]
    
    def _create_brick_zone(self, gp_bottom: List[int], gp_top: List[int], layer_name: str) -> Zone:
        """
        创建BRICK单元
        
        Args:
            gp_bottom: 底面4个节点ID [sw, se, ne, nw] (逆时针)
            gp_top: 顶面4个节点ID [sw, se, ne, nw] (逆时针)
            layer_name: 所属layer名称
        
        Returns:
            Zone对象
        """
        if len(gp_bottom) != 4 or len(gp_top) != 4:
            raise ValueError("BRICK requires 4 bottom + 4 top nodes")

        zone_id = self._next_zone_id
        self._next_zone_id += 1

        # FLAC3D B8 节点顺序(B方案): bottom[SW,SE,NW,NE] + top[SW,SE,NW,NE]
        sw, se, ne, nw = gp_bottom
        sw_t, se_t, ne_t, nw_t = gp_top
        gridpoint_ids = [
            sw,
            se,
            nw,
            ne,
            sw_t,
            se_t,
            nw_t,
            ne_t,
        ]

        return Zone(
            id=zone_id,
            type="brick",
            gridpoint_ids=gridpoint_ids,
            group=layer_name
        )
