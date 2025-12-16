# backend/z_section_slicer.py
"""
Z轴剖面切片功能 - 从3D地质模型中提取水平剖面
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from coal_seam_blocks.modeling import BlockModel


def extract_z_section(
    block_models: List[BlockModel],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_coordinate: float,
    sampling_step: int = 1
) -> Dict:
    """
    根据指定 z 坐标提取水平剖面
    
    Args:
        block_models: BlockModel 列表 (从底到顶排序)
        grid_x: X 轴网格坐标 (1D 数组)
        grid_y: Y 轴网格坐标 (1D 数组)
        z_coordinate: 剖面的 z 坐标值
        sampling_step: 采样步长 (默认1,即不降采样)
        
    Returns:
        包含剖面数据的字典
    """
    try:
        if not block_models:
            raise ValueError("block_models 为空")
        
        if len(grid_x) == 0 or len(grid_y) == 0:
            raise ValueError("grid_x 或 grid_y 为空")
        
        # 确保是 numpy 数组
        grid_x = np.asarray(grid_x)
        grid_y = np.asarray(grid_y)
        
        # 获取第一个模型的表面尺寸作为参考
        ref_model = block_models[0]
        
        # 验证模型数据
        if not hasattr(ref_model, 'top_surface') or not hasattr(ref_model, 'bottom_surface'):
            raise ValueError(f"模型 '{ref_model.name}' 缺少 top_surface 或 bottom_surface 属性")
        
        if ref_model.top_surface is None or ref_model.bottom_surface is None:
            raise ValueError(f"模型 '{ref_model.name}' 的表面数据为 None")
        
        model_ny, model_nx = ref_model.top_surface.shape
        
        print(f"[Z剖面] 网格尺寸: grid_x={len(grid_x)}, grid_y={len(grid_y)}")
        print(f"[Z剖面] 模型表面尺寸: ny={model_ny}, nx={model_nx}")
        
        # 使用模型表面的实际尺寸，而不是 grid_x/grid_y 的长度
        # 因为模型可能已经在建模时做了降采样
        ny, nx = model_ny, model_nx
        
        # 重新生成匹配模型尺寸的网格坐标
        x_min, x_max = float(grid_x.min()), float(grid_x.max())
        y_min, y_max = float(grid_y.min()), float(grid_y.max())
        
        grid_x_matched = np.linspace(x_min, x_max, nx)
        grid_y_matched = np.linspace(y_min, y_max, ny)
        
        # 创建网格
        XI, YI = np.meshgrid(grid_x_matched, grid_y_matched)
        
        # 展平坐标
        x_flat = XI.flatten()
        y_flat = YI.flatten()
        total_points = len(x_flat)
        
        # 初始化结果数组
        lithology_names = np.full(total_points, "", dtype=object)
        lithology_indices = np.full(total_points, -1, dtype=int)
        z_values = np.full(total_points, np.nan, dtype=float)
        
        # 计算模型的 z 范围
        all_tops = []
        all_bottoms = []
        
        for bm in block_models:
            if bm.top_surface is not None:
                all_tops.append(bm.top_surface.flatten())
            if bm.bottom_surface is not None:
                all_bottoms.append(bm.bottom_surface.flatten())
        
        if not all_tops or not all_bottoms:
            raise ValueError("所有模型的表面数据均为 None")
        
        all_tops_arr = np.concatenate(all_tops)
        all_bottoms_arr = np.concatenate(all_bottoms)
        z_min = float(np.nanmin(all_bottoms_arr))
        z_max = float(np.nanmax(all_tops_arr))
        
        print(f"\n[Z剖面] 提取 z={z_coordinate:.2f} 的剖面")
        print(f"[Z剖面] 模型范围: z_min={z_min:.2f}, z_max={z_max:.2f}")
        print(f"[Z剖面] 实际网格尺寸: {ny} x {nx} = {total_points} 点")
        
        # 检查 z_coordinate 是否在合理范围内
        if z_coordinate < z_min or z_coordinate > z_max:
            print(f"⚠️ [Z剖面] 警告: z={z_coordinate:.2f} 超出模型范围 [{z_min:.2f}, {z_max:.2f}]")
        
        # 为每个网格点确定所在岩层
        for layer_idx, model in enumerate(block_models):
            if model.top_surface is None or model.bottom_surface is None:
                print(f"⚠️ [Z剖面] 跳过模型 '{model.name}': 表面数据为 None")
                continue
            
            # 直接使用模型表面数据（已经是正确尺寸）
            top_flat = model.top_surface.flatten()
            bottom_flat = model.bottom_surface.flatten()
            
            # 确保尺寸匹配
            if len(top_flat) != total_points:
                print(f"⚠️ [Z剖面] 尺寸不匹配: 岩层'{model.name}' top_flat={len(top_flat)}, total_points={total_points}")
                continue
            
            # 找到 z_coordinate 位于该岩层内的点
            # bottom <= z_coordinate < top
            mask = (bottom_flat <= z_coordinate) & (z_coordinate < top_flat)
            
            # 标记这些点的岩性
            lithology_names[mask] = model.name
            lithology_indices[mask] = layer_idx
            z_values[mask] = z_coordinate
            
            n_hits = np.sum(mask)
            if n_hits > 0:
                print(f"[Z剖面] 岩层 '{model.name}' (索引={layer_idx}): {n_hits} 点")
        
        # 统计未分配岩性的点
        unassigned_mask = lithology_indices == -1
        n_unassigned = np.sum(unassigned_mask)
        
        if n_unassigned > 0:
            # 对于未分配的点,标记为"无数据"
            lithology_names[unassigned_mask] = "无数据"
            lithology_indices[unassigned_mask] = len(block_models)  # 使用特殊索引
            print(f"[Z剖面] 未分配岩性: {n_unassigned} 点 (可能在模型外)")
        
        # 构建图例
        legend = []
        unique_layers = {}
        
        for idx, model in enumerate(block_models):
            if np.any(lithology_indices == idx):
                unique_layers[idx] = model.name
                legend.append({
                    'index': idx,
                    'name': model.name,
                    'color': _get_layer_color(model.name, idx)
                })
        
        # 添加"无数据"图例
        if n_unassigned > 0:
            legend.append({
                'index': len(block_models),
                'name': '无数据',
                'color': '#CCCCCC'
            })
        
        print(f"[Z剖面] 图例包含 {len(legend)} 种岩性")
        
        # 将 lithology_index 重塑为 2D 网格，用于 heatmap 渲染
        lithology_grid = lithology_indices.reshape((ny, nx))
        
        # 清理数据：将 NaN/inf 替换为有效值，确保 JSON 序列化成功
        z_values_clean = np.where(np.isfinite(z_values), z_values, z_coordinate)
        
        return {
            'z_coordinate': float(z_coordinate),
            'x_coords': x_flat.tolist(),
            'y_coords': y_flat.tolist(),
            'lithology': lithology_names.tolist(),
            'lithology_index': lithology_indices.tolist(),
            'z_values': z_values_clean.tolist(),  # 使用清理后的值
            'legend': legend,
            'z_range': (float(z_min), float(z_max)),
            'grid_shape': (int(ny), int(nx)),
            # heatmap 渲染需要的网格数据 (使用匹配模型尺寸的坐标)
            'grid_x': grid_x_matched.tolist(),
            'grid_y': grid_y_matched.tolist(),
            'lithology_grid': lithology_grid.tolist()  # 2D 数组 [ny][nx]
        }
    
    except Exception as e:
        print(f"[Z剖面] ❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Z剖面提取失败: {str(e)}")


def get_z_range_from_models(block_models: List[BlockModel]) -> Tuple[float, float]:
    """
    获取模型的 z 范围
    
    Args:
        block_models: BlockModel 列表
        
    Returns:
        (z_min, z_max)
    """
    if not block_models:
        return (0.0, 0.0)
    
    all_tops = np.concatenate([bm.top_surface.flatten() for bm in block_models])
    all_bottoms = np.concatenate([bm.bottom_surface.flatten() for bm in block_models])
    
    z_min = float(np.nanmin(all_bottoms))
    z_max = float(np.nanmax(all_tops))
    
    return (z_min, z_max)


def _get_layer_color(layer_name: str, layer_index: int) -> str:
    """
    根据岩层名称和索引返回颜色
    
    优先根据岩性关键词匹配颜色,否则使用索引循环
    """
    name_lower = layer_name.lower()
    
    # 煤层 - 黑色系
    if '煤' in layer_name or 'coal' in name_lower:
        return '#1a1a1a'
    
    # 砂岩 - 黄色系
    if '砂' in layer_name or 'sand' in name_lower:
        return '#f4a460'
    
    # 泥岩 - 棕色系
    if '泥' in layer_name or 'mud' in name_lower or '页' in layer_name:
        return '#8b7355'
    
    # 灰岩 - 灰色系
    if '灰' in layer_name or 'lime' in name_lower:
        return '#a9a9a9'
    
    # 粉砂 - 浅黄色
    if '粉' in layer_name or 'silt' in name_lower:
        return '#deb887'
    
    # 页岩 - 深棕色
    if '页' in layer_name or 'shale' in name_lower:
        return '#654321'
    
    # 默认调色板 (使用索引循环)
    default_palette = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#546570'
    ]
    
    return default_palette[layer_index % len(default_palette)]
