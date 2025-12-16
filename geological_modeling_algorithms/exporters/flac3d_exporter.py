import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base_exporter import BaseExporter

class FLAC3DExporter(BaseExporter):
    """
    导出地质模型为 FLAC3D 命令脚本 (.dat)
    直接通过 program call 执行 zone gridpoint/zones 指令生成网格。
    """
    
    def export(self, data: Dict[str, Any], output_path: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        导出 FLAC3D 网格文件
        
        Args:
            data: 包含地层数据的字典，格式如下:
                  {
                      "layers": [
                          {
                              "name": "LayerName",
                              "grid_x": np.ndarray,
                              "grid_y": np.ndarray,
                              "grid_z": np.ndarray,       # 顶板
                              "grid_z_bottom": np.ndarray # 底板 (可选，如果没有则尝试用 thickness)
                              "thickness": np.ndarray     # 厚度 (可选)
                          },
                          ...
                      ]
                  }
            output_path: 输出文件路径
            options: 导出选项
            
        Returns:
            str: 导出文件的路径
        """
        options = options or {}

        desired_ext = options.get("extension") or Path(output_path).suffix
        if desired_ext:
            desired_ext = desired_ext if desired_ext.startswith('.') else f".{desired_ext}"
        else:
            desired_ext = '.dat'
        output_path = str(Path(output_path).with_suffix(desired_ext))
        
        layers = data.get("layers", [])
        if not layers:
            raise ValueError("没有可导出的地层数据")
        
        print(f"[FLAC3D Export] 开始导出 {len(layers)} 个地层到: {output_path}")
            
        # 收集所有节点和单元
        nodes = [] # (id, x, y, z)
        zones = [] # (type, id, [node_indices], group_name)
        
        node_map = {} # (x, y, z) -> node_id
        next_node_id = 1
        next_zone_id = 1
        
        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get("name", "Default")
            # 替换非法字符
            safe_layer_name = "".join(c for c in layer_name if c.isalnum() or c in "_-")
            
            grid_x = layer.get("grid_x")
            grid_y = layer.get("grid_y")
            grid_z_top = layer.get("top_surface_z")
            if grid_z_top is None:
                grid_z_top = layer.get("grid_z")
            
            if grid_x is None or grid_y is None or grid_z_top is None:
                continue
                
            grid_x = np.array(grid_x)
            grid_y = np.array(grid_y)
            grid_z_top = np.array(grid_z_top)
            
            if grid_x.ndim == 1 and grid_y.ndim == 1:
                 grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            
            grid_z_bottom = layer.get("bottom_surface_z")
            if grid_z_bottom is None:
                grid_z_bottom = layer.get("grid_z_bottom")
            if grid_z_bottom is None:
                thickness = layer.get("thickness")
                if thickness is not None:
                    thickness = np.array(thickness)
                    grid_z_bottom = grid_z_top - thickness
                    print(f"  [Layer {layer_idx + 1}] {safe_layer_name}: 使用厚度计算底面")
                else:
                    # 无法生成体单元
                    print(f"  [Layer {layer_idx + 1}] {safe_layer_name}: 跳过(缺少底面或厚度数据)")
                    continue
            else:
                grid_z_bottom = np.array(grid_z_bottom)
                print(f"  [Layer {layer_idx + 1}] {safe_layer_name}: 使用底面数据")
            
            rows, cols = grid_z_top.shape
            print(f"  [Layer {layer_idx + 1}] {safe_layer_name}: 网格尺寸 {rows}x{cols}")
            
            # 性能优化：预先为所有节点创建唯一ID映射
            # 使用向量化操作而不是循环
            layer_zones = 0
            
            # 预先检查有效性（避免在循环中重复检查）
            valid_bottom = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z_bottom))
            valid_top = ~(np.isnan(grid_x) | np.isnan(grid_y) | np.isnan(grid_z_top))
            
            # 为底面和顶面创建节点（使用集合去重）
            for surface_name, grid_z, valid_mask in [
                ('bottom', grid_z_bottom, valid_bottom),
                ('top', grid_z_top, valid_top)
            ]:
                # 只处理有效的节点
                valid_indices = np.argwhere(valid_mask)
                for idx in valid_indices:
                    r, c = idx
                    x = float(grid_x[r, c])
                    y = float(grid_y[r, c])
                    z = float(grid_z[r, c])
                    
                    key = (round(x, 4), round(y, 4), round(z, 4))
                    if key not in node_map:
                        node_map[key] = next_node_id
                        nodes.append((next_node_id, x, y, z))
                        next_node_id += 1
            
            # 快速生成单元（现在所有节点ID都已知）
            for r in range(rows - 1):
                for c in range(cols - 1):
                    # 检查这个单元的所有8个节点是否都有效
                    cell_valid = (
                        valid_bottom[r, c] and valid_bottom[r, c+1] and 
                        valid_bottom[r+1, c+1] and valid_bottom[r+1, c] and
                        valid_top[r, c] and valid_top[r, c+1] and 
                        valid_top[r+1, c+1] and valid_top[r+1, c]
                    )
                    
                    if not cell_valid:
                        continue
                    
                    # 获取8个节点ID（顺序：底面4个 + 顶面4个）
                    cell_nodes = []
                    indices = [(r, c), (r, c+1), (r+1, c+1), (r+1, c)]
                    
                    # 底面4个节点
                    for rr, cc in indices:
                        x = float(grid_x[rr, cc])
                        y = float(grid_y[rr, cc])
                        z = float(grid_z_bottom[rr, cc])
                        key = (round(x, 4), round(y, 4), round(z, 4))
                        cell_nodes.append(node_map[key])
                    
                    # 顶面4个节点
                    for rr, cc in indices:
                        x = float(grid_x[rr, cc])
                        y = float(grid_y[rr, cc])
                        z = float(grid_z_top[rr, cc])
                        key = (round(x, 4), round(y, 4), round(z, 4))
                        cell_nodes.append(node_map[key])
                    
                    zones.append(('B8', next_zone_id, cell_nodes, safe_layer_name))
                    next_zone_id += 1
                    layer_zones += 1
            
            print(f"  [Layer {layer_idx + 1}] 生成了 {layer_zones} 个体单元 (优化算法)")

        print(f"[FLAC3D Export] 总共生成 {len(nodes)} 个节点, {len(zones)} 个体单元")
        
        if len(zones) == 0:
            raise ValueError("未能生成任何有效的体单元，请检查数据是否包含厚度信息")

        coord_offset = options.get("coordinate_offset")
        if coord_offset is None:
            coord_offset = options.get("coord_offset")

        normalize_coords = bool(options.get("normalize_coords"))
        x_off = y_off = z_off = 0.0

        if coord_offset is not None:
            if len(coord_offset) != 3:
                raise ValueError("coordinate_offset 必须是长度为3的(x,y,z)序列")
            x_off, y_off, z_off = map(float, coord_offset)
            print(f"[FLAC3D Export] 使用外部坐标偏移: ({x_off}, {y_off}, {z_off})")
        elif normalize_coords and nodes:
            nodes_array = np.array([(n[1], n[2], n[3]) for n in nodes])
            centroid = np.mean(nodes_array, axis=0)
            x_off, y_off, z_off = centroid
            print(f"[FLAC3D Export] normalize_coords=True, 使用模型质心偏移 {centroid}")

        # 写入文件（使用缓冲提高性能）
        print(f"[FLAC3D Export] 开始写入文件...")
        lines = []
        
        # 文件头（使用FLAC3D命令格式）
        lines.append("; FLAC3D Grid Import Script\n")
        lines.append(f"; Total Gridpoints: {len(nodes)}\n")
        lines.append(f"; Total Zones: {len(zones)}\n")
        lines.append("\n")
        lines.append("model new\n")
        lines.append("model large-strain off\n")
        lines.append("\n")
        
        # 创建节点映射表（FLAC3D会自动分配ID）
        print(f"[FLAC3D Export] 开始格式化节点...")
        lines.append("; Creating gridpoints...\n")
        node_map = {}  # 原始ID -> FLAC3D顺序ID的映射
        for idx, (nid, x, y, z) in enumerate(nodes, start=1):
            nx = x - x_off
            ny = y - y_off
            nz = z - z_off
            lines.append(f"zone gridpoint create position ({nx:.4f},{ny:.4f},{nz:.4f})\n")
            node_map[nid] = idx
        
        print(f"[FLAC3D Export] 节点数据已格式化 ({len(nodes)} 个节点)")
        
        lines.append("\n")
        lines.append("; Creating zones...\n")
        
        # 创建单元（使用zone.create命令，映射节点ID）
        print(f"[FLAC3D Export] 开始格式化单元...")
        for ztype, zid, nids, group in zones:
            if len(nids) == 8:
                # 将原始节点ID映射为FLAC3D的顺序ID
                mapped_nids = [node_map[nid] for nid in nids]
                gp_str = " ".join(map(str, mapped_nids))
                lines.append(f"zone create brick gridpoint {gp_str} group '{group}'\n")
        
        print(f"[FLAC3D Export] 单元数据已格式化 ({len(zones)} 个单元)")
        
        # 分组信息（作为注释记录）
        lines.append("\n")
        lines.append("; End of FLAC3D grid definition\n")
        
        # 一次性写入（比逐行写入快得多）
        print(f"[FLAC3D Export] 正在写入文件到磁盘...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"[FLAC3D Export] 文件已保存: {output_path}")
        return output_path
