import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter


class BlockModel:
    def __init__(self,
                 name: str,
                 points: int,
                 top_surface: np.ndarray,
                 bottom_surface: np.ndarray):
        self.name = name
        self.points = points
        self.top_surface = np.asarray(top_surface, dtype=float)
        self.bottom_surface = np.asarray(bottom_surface, dtype=float)

        # 计算厚度网格及统计信息
        thickness = self.top_surface - self.bottom_surface
        thickness = np.clip(thickness, 0.0, None)
        self.thickness_grid = thickness

        def _safe_stat(func, array, default=0.0):
            try:
                value = func(array)
                if np.isfinite(value):
                    return float(value)
            except ValueError:
                pass
            return float(default)

        self.avg_thickness = _safe_stat(np.nanmean, thickness)
        self.max_thickness = _safe_stat(np.nanmax, thickness)
        self.avg_height = _safe_stat(np.nanmean, self.top_surface)
        self.max_height = _safe_stat(np.nanmax, self.top_surface)
        self.min_height = _safe_stat(np.nanmin, self.top_surface)
        self.avg_bottom = _safe_stat(np.nanmean, self.bottom_surface)
        self.min_bottom = _safe_stat(np.nanmin, self.bottom_surface)
        # 兼容早期版本仍读取 base 属性的场景
        self.base = self.avg_bottom


def build_grids(x_values: np.ndarray, y_values: np.ndarray, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(x_values.min(), x_values.max(), resolution)
    yi = np.linspace(y_values.min(), y_values.max(), resolution)
    XI, YI = np.meshgrid(xi, yi)
    xi_flat = XI.flatten()
    yi_flat = YI.flatten()
    return XI, YI, xi_flat, yi_flat


def interpolate_seam(x_points: np.ndarray, y_points: np.ndarray, thickness: np.ndarray,
                      xi: np.ndarray, yi: np.ndarray, method_callable) -> np.ndarray:
    values = method_callable(x_points, y_points, thickness, xi, yi)
    values = values.reshape((int(np.sqrt(len(xi))), int(np.sqrt(len(xi)))))
    
    # ⚠️ 关键修复: NaN/Inf不能转0,会导致厚度为0!
    # 用厚度数据的中位数填充
    invalid_mask = ~np.isfinite(values)
    if np.any(invalid_mask):
        fill_value = float(np.median(thickness)) if len(thickness) > 0 else 1.0
        values = np.where(np.isfinite(values), values, fill_value)
    
    return np.clip(values, 0.0, None)


def check_vertical_order(block_models: List[BlockModel]) -> Dict[str, int]:
    """
    检查相邻层在每个网格点的垂向顺序
    
    检查相邻层是否存在 upper.bottom < lower.top 的情况(即重叠)
    
    Args:
        block_models: BlockModel列表,应为从底到顶排序
        
    Returns:
        包含检查结果的字典
    """
    if not block_models:
        print("[check_vertical_order] 无 block_models")
        return {}
    
    nlay = len(block_models)
    if nlay < 2:
        print("[check_vertical_order] 只有1层,无需检查")
        return {}
    
    # 堆叠所有层的底面和顶面 (nlay, ny, nx)
    bottoms = np.stack([bm.bottom_surface for bm in block_models])
    tops = np.stack([bm.top_surface for bm in block_models])
    
    ny, nx = bottoms.shape[1:]
    total_cells = ny * nx
    
    print(f"\n[垂向顺序检查] 开始检查 {nlay} 层，总网格点: {total_cells} ({ny}×{nx})")
    print(f"{'':>4} {'下层':>15} {'上层':>15} {'重叠点数':>10} {'重叠比例':>10} {'最大重叠':>12}")
    print("-" * 80)
    
    total_bad = 0
    results = {}
    
    for k in range(nlay - 1):
        lower_top = tops[k]
        upper_bottom = bottoms[k + 1]
        
        # 只在有效点检查
        valid = np.isfinite(lower_top) & np.isfinite(upper_bottom)
        bad = valid & (upper_bottom < lower_top)
        
        bad_count = int(bad.sum())
        valid_count = int(valid.sum())
        
        lower_name = block_models[k].name
        upper_name = block_models[k + 1].name
        
        if valid_count > 0:
            bad_percent = (bad_count / valid_count) * 100
            
            # 计算最大重叠量
            overlap = np.where(bad, lower_top - upper_bottom, 0.0)
            max_overlap = float(np.max(overlap)) if bad_count > 0 else 0.0
            
            status = "❌" if bad_count > 0 else "✅"
            print(f"{status} {k:>2} {lower_name:>15} {upper_name:>15} {bad_count:>10} {bad_percent:>9.1f}% {max_overlap:>11.2f}m")
            
            total_bad += bad_count
            results[f"{lower_name}→{upper_name}"] = {
                'bad_count': bad_count,
                'total_count': valid_count,
                'max_overlap': max_overlap
            }
        else:
            print(f"⚠️ {k:>2} {lower_name:>15} {upper_name:>15} {'无有效点':>10}")
    
    print("-" * 80)
    if total_bad == 0:
        print(f"✅ 检查通过: 所有相邻层在所有网格点都满足垂向顺序")
    else:
        print(f"❌ 检查失败: 共 {total_bad} 个网格点存在层间重叠")
    print()
    
    return results


def enforce_columnwise_order(block_models: List[BlockModel], 
                            min_gap: float = 0.5, 
                            min_thickness: float = 0.5) -> None:
    """
    对每个(y,x)垂直柱子强制重排层序
    
    逐列处理,按bottom深度从小到大排序,然后自下而上重新码放,
    保证相邻层之间有min_gap,每层厚度不小于min_thickness。
    
    Args:
        block_models: BlockModel列表,会直接修改其bottom_surface和top_surface
        min_gap: 最小层间间隙(米),默认0.5米
        min_thickness: 最小层厚(米),默认0.5米
    """
    if not block_models:
        return
    
    nlay = len(block_models)
    if nlay < 2:
        return
    
    print(f"\n[逐列排序] 开始对 {nlay} 层进行逐列垂向排序")
    print(f"           最小间隙: {min_gap}m, 最小厚度: {min_thickness}m")
    
    # 堆叠所有层 (nlay, ny, nx)
    bottoms = np.stack([bm.bottom_surface for bm in block_models])
    tops = np.stack([bm.top_surface for bm in block_models])
    
    ny, nx = bottoms.shape[1:]
    total_cells = ny * nx
    fixed_count = 0
    
    # 逐列处理
    for j in range(ny):
        for i in range(nx):
            # 提取这一列的所有层
            bcol = bottoms[:, j, i]
            tcol = tops[:, j, i]
            
            # 找出有效的层(bottom和top都是有限值)
            valid_idx = np.where(np.isfinite(bcol) & np.isfinite(tcol))[0]
            if valid_idx.size == 0:
                continue
            
            # 按原始bottom深度排序(从浅到深,即从下到上)
            order = valid_idx[np.argsort(bcol[valid_idx])]
            
            # 检查是否需要修复
            needs_fix = False
            for ii in range(len(order) - 1):
                if tops[order[ii], j, i] + min_gap > bottoms[order[ii+1], j, i]:
                    needs_fix = True
                    break
            
            if not needs_fix:
                continue
            
            fixed_count += 1
            
            # 这一列最底部的起始深度
            z_cur = float(np.min(bcol[valid_idx]))
            
            # 自下而上重新码放
            for idx in order:
                # 计算厚度
                thick = float(tcol[idx] - bcol[idx])
                if not np.isfinite(thick) or thick < min_thickness:
                    thick = min_thickness
                
                # 重新设置底面和顶面
                bottoms[idx, j, i] = z_cur
                tops[idx, j, i] = z_cur + thick
                
                # 更新下一层的起始位置
                z_cur = tops[idx, j, i] + float(min_gap)
    
    # 写回到BlockModel
    for k, bm in enumerate(block_models):
        bm.bottom_surface = bottoms[k]
        bm.top_surface = tops[k]
        bm.thickness_grid = tops[k] - bottoms[k]
    
    print(f"[逐列排序] 完成! 共修复 {fixed_count}/{total_cells} 个垂直柱 ({fixed_count/total_cells*100:.1f}%)\n")


def build_block_models(merged_df: pd.DataFrame,
                       seam_column: str,
                       x_col: str,
                       y_col: str,
                       thickness_col: str,
                       selected_seams: List[str],
                       method_callable,
                       resolution: int,
                       base_level: float,
                       gap_value: float) -> Tuple[List[BlockModel], List[str], Tuple[np.ndarray, np.ndarray]]:
    if merged_df.empty:
        raise ValueError("合并数据为空，无法建模")

    required_cols = [x_col, y_col, thickness_col, seam_column]
    valid_data = merged_df.dropna(subset=required_cols).copy()
    if len(valid_data) < 8:
        raise ValueError("生成块体至少需要8个有效数据点")

    valid_data[seam_column] = valid_data[seam_column].astype(str)
    x_vals = valid_data[x_col].astype(float)
    y_vals = valid_data[y_col].astype(float)

    if x_vals.nunique() < 2 or y_vals.nunique() < 2:
        raise ValueError("X 或 Y 坐标取值过少，无法构建网格")

    XI, YI, xi_flat, yi_flat = build_grids(x_vals.values, y_vals.values, resolution)

    block_models: List[BlockModel] = []
    skipped: List[str] = []
    current_base_surface = np.full((XI.shape[0], XI.shape[1]), float(base_level), dtype=float)

    for seam_name in selected_seams:
        seam_df = valid_data[valid_data[seam_column] == str(seam_name)]
        if seam_df.empty:
            skipped.append(f"{seam_name} (无数据点)")
            continue
        
        # 降低最小点数要求: 1个点也可以建模(使用最近邻插值)
        num_points = len(seam_df)
        if num_points < 1:
            skipped.append(f"{seam_name} (有效点 0)")
            continue

        x_points = seam_df[x_col].astype(float).values
        y_points = seam_df[y_col].astype(float).values
        thickness_points = pd.to_numeric(seam_df[thickness_col], errors='coerce').values
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(thickness_points)
        if not np.any(valid_mask):
            skipped.append(f"{seam_name} (厚度数据全部无效)")
            continue
        
        x_points = x_points[valid_mask]
        y_points = y_points[valid_mask]
        thickness_points = thickness_points[valid_mask]
        num_valid = len(thickness_points)
        
        # 🔍 诊断日志
        print(f"    [建模] {seam_name}: {num_valid}个有效厚度采样点")
        if num_valid > 0:
            print(f"           厚度范围: [{np.min(thickness_points):.2f}, {np.max(thickness_points):.2f}]m")
            print(f"           平均厚度: {np.mean(thickness_points):.2f}m")
            print(f"           中位数厚度: {np.median(thickness_points):.2f}m")
        
        # ⚠️ 最小点数要求提高到3,避免插值外推产生极端值
        if num_valid < 3:
            skipped.append(f"{seam_name} (有效点太少: {num_valid} < 3)")
            print(f"    [警告] {seam_name} 采样点不足3个,跳过建模")
            continue

        try:
            interpolated = method_callable(x_points, y_points, thickness_points, xi_flat, yi_flat)
            if interpolated is None:
                skipped.append(f"{seam_name} (插值无结果, {num_valid}个点)")
                continue

            thickness_grid = interpolated.reshape(XI.shape)
            thickness_grid = np.asarray(thickness_grid, dtype=float)
            if not np.isfinite(thickness_grid).any():
                skipped.append(f"{seam_name} (插值结果全为无效值, {num_valid}个点)")
                continue

            # ⚠️ 关键修复：厚度NaN不能转成0，会导致层间重叠！
            # 原因：厚度=0 → 顶面=底面 → 与下一层重合
            # 解决：用该层的平均厚度或最小有效厚度填充
            nan_count = np.isnan(thickness_grid).sum()
            if nan_count > 0:
                valid_thickness = thickness_grid[~np.isnan(thickness_grid)]
                if len(valid_thickness) > 0:
                    # 使用中位数填充（比平均值更稳健）
                    fill_value = float(np.median(valid_thickness))
                    # 确保填充值不小于最小有效厚度的一半
                    min_thickness = max(0.5, float(np.min(valid_thickness)) * 0.5)
                    fill_value = max(fill_value, min_thickness)
                else:
                    # 如果没有有效值，使用经验最小厚度
                    fill_value = 1.0  # 默认1米
                
                thickness_grid = np.nan_to_num(thickness_grid, nan=fill_value)
                print(f"    [建模] {seam_name}: {nan_count}个位置厚度缺失，用{fill_value:.2f}m填充")
            
            # ⚠️ 处理Inf和负值 - 不能转为0！转为NaN后用median填充
            inf_count = np.sum(np.isinf(thickness_grid)) + np.sum(thickness_grid < 0)
            if inf_count > 0:
                # 将Inf和负值标记为NaN
                thickness_grid = np.where(
                    np.isfinite(thickness_grid) & (thickness_grid >= 0),
                    thickness_grid,
                    np.nan
                )
                # 用中位数填充
                valid_thickness = thickness_grid[~np.isnan(thickness_grid)]
                if len(valid_thickness) > 0:
                    fill_value_inf = float(np.median(valid_thickness))
                else:
                    fill_value_inf = 1.0
                thickness_grid = np.nan_to_num(thickness_grid, nan=fill_value_inf)
                print(f"    [建模] {seam_name}: {inf_count}个无效值(Inf/负值)用{fill_value_inf:.2f}m填充")
            
            # 确保非负,并设置最小厚度(0.5m)防止退化几何体
            # 原因: 厚度为0会导致顶面=底面,生成STL时产生重叠的退化三角面片
            MIN_LAYER_THICKNESS = 0.5  # 最小层厚0.5米
            thickness_grid = np.clip(thickness_grid, MIN_LAYER_THICKNESS, None)
            
            zero_thickness_count = np.sum(thickness_grid == MIN_LAYER_THICKNESS)
            if zero_thickness_count > 0:
                total_cells = thickness_grid.size
                print(f"    [建模] {seam_name}: {zero_thickness_count}个位置厚度过小(<0.5m),已调整为{MIN_LAYER_THICKNESS}m ({zero_thickness_count/total_cells*100:.1f}%)")

            # 🔧 使用current_base_surface作为本层底面,自然实现自下而上堆叠
            bottom_surface = current_base_surface.copy()
            top_surface = bottom_surface + thickness_grid
            
            # 🔧 简单验证: 确保本层内部top >= bottom (理论上不会违反,这里仅作兜底)
            top_surface = np.maximum(top_surface, bottom_surface + MIN_LAYER_THICKNESS)
            
            # 🔧 使用current_base_surface作为本层底面,自然实现自下而上堆叠
            bottom_surface = current_base_surface.copy()
            top_surface = bottom_surface + thickness_grid
            
            # 🔧 简单验证: 确保本层内部top >= bottom (理论上不会违反,这里仅作兜底)
            top_surface = np.maximum(top_surface, bottom_surface + MIN_LAYER_THICKNESS)
            
            # 最终验证并记录 - 添加调试日志以便验证Z范围
            final_top_min = float(np.min(top_surface))
            final_top_max = float(np.max(top_surface))
            final_bottom_min = float(np.min(bottom_surface))
            final_bottom_max = float(np.max(bottom_surface))
            final_thickness_min = float(np.min(thickness_grid))
            final_thickness_max = float(np.max(thickness_grid))
            
            print(f"    [最终] {seam_name} 建模完成")
            print(f"           底面Z: [{final_bottom_min:.2f}, {final_bottom_max:.2f}]m (极差: {final_bottom_max - final_bottom_min:.2f}m)")
            print(f"           厚度:  [{final_thickness_min:.2f}, {final_thickness_max:.2f}]m (极差: {final_thickness_max - final_thickness_min:.2f}m)")
            print(f"           顶面Z: [{final_top_min:.2f}, {final_top_max:.2f}]m (极差: {final_top_max - final_top_min:.2f}m)")
            
            block_models.append(BlockModel(
                name=str(seam_name),
                points=num_valid,
                top_surface=top_surface,
                bottom_surface=bottom_surface
            ))

            # 更新下一层的基准面: current_base_surface = 本层顶面 + gap
            # 这样下一层的底面自然从本层顶面之上开始,实现严格自下而上堆叠
            current_base_surface = top_surface
            if gap_value:
                current_base_surface = current_base_surface + float(gap_value)
                next_bottom_mean = float(np.mean(current_base_surface))
                print(f"           [层间] 添加间隙 {float(gap_value):.2f}m, 下一层底面平均高程: {next_bottom_mean:.2f}m")
        
        except Exception as e:
            skipped.append(f"{seam_name} (插值失败: {str(e)[:30]}, {num_valid}个点)")
            continue

    if not block_models:
        raise RuntimeError("选定的岩层数据不足以生成模型")

    return block_models, skipped, (XI, YI)
