import os
import json
import zipfile
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from .stl_exporter import STLExporter

try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


class LayeredSTLExporter:
    """
    分层STL导出器
    
    将地质模型的每一层导出为独立的STL文件，并打包成ZIP
    适用于FLAC3D逐层导入，避免内部分层面导致的网格生成失败
    
    功能：
    1. 每层导出为单独的STL文件
    2. 生成manifest.json元数据文件
    3. 生成FLAC3D导入脚本（.fish）
    4. 生成README导入说明
    5. 自动打包为ZIP文件
    """
    
    def __init__(self):
        self.stl_exporter = STLExporter()
        self.temp_files = []
    
    def _fix_layer_overlap(self, layers: List[Dict], min_gap: float = 0.5, min_thickness: float = 0.5):
        """
        检测并修复层间重叠问题 - 按修改建议优化版本
        
        Args:
            layers: 地层列表（应为从底到顶排列）
            min_gap: 最小层间间隙（米），默认0.5米
            min_thickness: 最小层厚（米），默认0.5米
        """
        import numpy as np
        
        if not layers:
            return
        
        # 🔧 关键修复1: 确保layers是从底到顶排序（按bottom_surface平均值）
        print(f"[Fix Overlap] 开始修复层间重叠，共{len(layers)}层")
        
        # 首先按底面平均高程排序
        layers.sort(key=lambda l: float(np.nanmean(
            l.get('bottom_surface') if 'bottom_surface' in l else 
            l.get('grid_z_bottom') if 'grid_z_bottom' in l else 
            l.get('bottom_surface_z', 0)
        )))
        
        print(f"[Fix Overlap] 已按底面高程排序（从底到顶）")
        
        # 🔧 关键修复2: 逐点计算需要抬升量，使用全场最大值
        for idx in range(len(layers) - 1):
            lower = layers[idx]
            upper = layers[idx + 1]
            
            # 获取下层顶面
            lower_top = lower.get('top_surface')
            if lower_top is None:
                lower_top = lower.get('grid_z')
            if lower_top is None:
                lower_top = lower.get('top_surface_z')
            
            # 获取上层底面
            upper_bottom = upper.get('bottom_surface')
            if upper_bottom is None:
                upper_bottom = upper.get('grid_z_bottom')
            if upper_bottom is None:
                upper_bottom = upper.get('bottom_surface_z')
            
            if lower_top is None or upper_bottom is None:
                continue
            
            lower_top = np.asarray(lower_top, dtype=float)
            upper_bottom = np.asarray(upper_bottom, dtype=float)
            
            # 要求: upper_bottom >= lower_top + min_gap
            required_bottom = lower_top + float(min_gap)
            
            # 逐点计算需要抬升多少
            delta = required_bottom - upper_bottom
            # 只关心"需要抬高"的地方，其余置0
            delta = np.where(delta > 0.0, delta, 0.0)
            
            # 统一采用"全场最大需要抬升量"来抬这一层，避免层内扭曲
            adjust = float(np.nanmax(delta))
            
            lower_name = lower.get('name', f'Layer_{idx}')
            upper_name = upper.get('name', f'Layer_{idx+1}')
            
            if adjust <= 0:
                print(f"  [{lower_name} → {upper_name}] 无需调整，间隙充足")
                continue  # 不需要调整
            
            print(f"  [{lower_name} → {upper_name}] 检测到重叠，需要上抬 {adjust:.3f}m")
            
            # 抬升上层的底面和顶面
            upper_bottom_new = upper_bottom + adjust
            
            if 'bottom_surface' in upper:
                upper['bottom_surface'] = upper_bottom_new
            if 'grid_z_bottom' in upper:
                upper['grid_z_bottom'] = upper_bottom_new
            if 'bottom_surface_z' in upper:
                upper['bottom_surface_z'] = upper_bottom_new
            
            # 同时抬升顶面
            upper_top = upper.get('top_surface')
            if upper_top is None:
                upper_top = upper.get('grid_z')
            if upper_top is None:
                upper_top = upper.get('top_surface_z')
            
            if upper_top is not None:
                upper_top = np.asarray(upper_top, dtype=float) + adjust
                if 'top_surface' in upper:
                    upper['top_surface'] = upper_top
                if 'grid_z' in upper:
                    upper['grid_z'] = upper_top
                if 'top_surface_z' in upper:
                    upper['top_surface_z'] = upper_top
            
            # 🔧 关键修复3: 兜底保证上层内部厚度不小于min_thickness
            upper_top_final = upper.get('top_surface', upper.get('grid_z', upper.get('top_surface_z')))
            upper_bottom_final = upper.get('bottom_surface', upper.get('grid_z_bottom', upper.get('bottom_surface_z')))
            
            if upper_top_final is not None and upper_bottom_final is not None:
                upper_top_final = np.asarray(upper_top_final, dtype=float)
                upper_bottom_final = np.asarray(upper_bottom_final, dtype=float)
                
                # 确保每个位置厚度 >= min_thickness
                upper_top_final = np.maximum(
                    upper_top_final,
                    upper_bottom_final + float(min_thickness)
                )
                
                if 'top_surface' in upper:
                    upper['top_surface'] = upper_top_final
                if 'grid_z' in upper:
                    upper['grid_z'] = upper_top_final
                if 'top_surface_z' in upper:
                    upper['top_surface_z'] = upper_top_final
            
            # 输出调整后的Z范围
            final_lower_top_min = float(np.nanmin(lower_top))
            final_lower_top_max = float(np.nanmax(lower_top))
            final_upper_bottom_min = float(np.nanmin(upper_bottom_new))
            final_upper_bottom_max = float(np.nanmax(upper_bottom_new))
            actual_gap_min = final_upper_bottom_min - final_lower_top_max
            
            print(f"    调整后: 下层顶面 [{final_lower_top_min:.2f}, {final_lower_top_max:.2f}]m")
            print(f"            上层底面 [{final_upper_bottom_min:.2f}, {final_upper_bottom_max:.2f}]m")
            print(f"            实际最小间隙: {actual_gap_min:.3f}m")
        
        print(f"[Fix Overlap] 层间重叠修复完成")
    
    def _create_top_plate_layer(self, layers: List[Dict], top_plate_thickness: float = 10.0) -> Dict:
        """
        创建顶板层,填平最顶层的曲面
        
        Args:
            layers: 地层列表
            top_plate_thickness: 顶板厚度(m),默认10m
        
        Returns:
            顶板层数据字典
        """
        import numpy as np
        
        if not layers:
            raise ValueError("无法创建顶板:地层列表为空")
        
        # 找到最顶层(地层列表是从下到上排列,最后一个是最顶层)
        top_layer = layers[-1]
        
        # 获取顶层的顶面数据
        grid_x = np.array(top_layer.get("grid_x"))
        grid_y = np.array(top_layer.get("grid_y"))
        
        # 获取顶层的顶面高程
        if "top_surface_z" in top_layer:
            top_surface_z = np.array(top_layer["top_surface_z"])
        elif "grid_z" in top_layer:
            top_surface_z = np.array(top_layer["grid_z"])
        else:
            raise ValueError("最顶层缺少高程数据")
        
        # 找到最高点
        max_z = np.nanmax(top_surface_z)
        
        print(f"  [顶板生成] 最顶层: {top_layer.get('name', '未命名')}")
        print(f"  [顶板生成] 最高点: {max_z:.2f}m")
        print(f"  [顶板生成] 顶板厚度: {top_plate_thickness:.2f}m")
        
        # 创建顶板层:
        # - 底面: 跟随最顶层曲面
        # - 顶面: 统一平面,高度为最高点+顶板厚度
        top_plate_layer = {
            "name": "顶板",
            "grid_x": grid_x.copy(),
            "grid_y": grid_y.copy(),
            "bottom_surface_z": top_surface_z.copy(),  # 底面跟随曲面
            "top_surface_z": np.full_like(top_surface_z, max_z + top_plate_thickness)  # 顶面平坦
        }
        
        print(f"  [顶板生成] 底面高程范围: [{np.nanmin(top_surface_z):.2f}, {max_z:.2f}]m")
        print(f"  [顶板生成] 顶面高程: {max_z + top_plate_thickness:.2f}m (平面)")
        
        return top_plate_layer
    
    def export_layered(self, data: Dict[str, Any], output_zip_path: str,
                      options: Optional[Dict[str, Any]] = None) -> str:
        """
        分层导出地质模型为多个STL文件并打包
        
        Args:
            data: 包含地层数据的字典
            output_zip_path: 输出ZIP文件路径
            options: 导出选项
                - format: 'binary' 或 'ascii'
                - downsample_factor: 降采样倍数
                - normalize_coords: 是否坐标归一化
                - include_fish_script: 是否生成FISH脚本(默认True)
                - add_top_plate: 是否添加顶板层(默认True)
                - top_plate_thickness: 顶板厚度(m,默认10m)
        
        Returns:
            str: 输出ZIP文件的路径
        """
        if options is None:
            options = {}
        
        layers = data.get("layers", [])
        if not layers:
            raise ValueError("没有可导出的地层数据")
        
        print(f"[Layered STL Export] 开始处理 {len(layers)} 个地层")
        
        # 🔧 步骤1: 先修复层间重叠（在添加顶板之前）
        print("  [步骤1] 检查并修复层间重叠...")
        min_gap = options.get("min_layer_gap", 0.5)
        min_thickness = options.get("min_layer_thickness", 0.5)
        self._fix_layer_overlap(layers, min_gap=min_gap, min_thickness=min_thickness)
        
        # 🔧 步骤2: 添加顶板层（在修复重叠之后）
        add_top_plate = options.get("add_top_plate", True)
        if add_top_plate:
            top_plate_thickness = options.get("top_plate_thickness", 10.0)
            print(f"  [步骤2] 添加顶板层 (厚度: {top_plate_thickness}m)")
            top_plate_layer = self._create_top_plate_layer(layers, top_plate_thickness)
            # 将顶板追加到列表末尾(因为地层是从下到上排列)
            layers.append(top_plate_layer)
            # 更新data中的layers引用
            data["layers"] = layers
        
        print(f"[Layered STL Export] 准备导出 {len(layers)} 个地层（含顶板）")
        
        # 创建临时目录
        temp_dir = os.path.join(os.path.dirname(output_zip_path), "_temp_stl_export")
        os.makedirs(temp_dir, exist_ok=True)
        
        exported_files = []
        min_layer_gap = options.get("min_layer_gap", 0.5)
        manifest_data = {
            "export_time": datetime.now().isoformat(),
            "total_layers": len(layers),
            "format": options.get("format", "binary"),
            "downsample_factor": options.get("downsample_factor", 5),
            "coordinate_normalized": options.get("normalize_coords", True),
            "min_layer_gap": min_layer_gap,
            "quality_checks_enabled": True,
            "layers": []
        }
        
        # 预先计算全局坐标偏移量（确保所有层使用相同的坐标系）
        global_offset = None
        if options.get("normalize_coords", True):
            print("  [步骤3] 计算全局坐标偏移量...")
            global_offset = self.stl_exporter._calculate_coord_offset(layers, True)
            print(f"  [全局偏移] X={global_offset[0]:.2f}, Y={global_offset[1]:.2f}, Z={global_offset[2]:.2f}")
        
        # 🔧 步骤4: 逐层导出STL文件
        for layer_idx, layer in enumerate(layers):
            layer_name = layer.get("name", f"Layer_{layer_idx}")
            # 转换为英文文件名（FLAC3D对中文支持不好）
            english_name = self._to_english_filename(layer_name)
            
            # 文件名格式: 01_coal_6.stl, 02_sandy_mudstone.stl
            stl_filename = f"{layer_idx+1:02d}_{english_name}.stl"
            stl_filepath = os.path.join(temp_dir, stl_filename)
            
            print(f"  [{layer_idx+1}/{len(layers)}] 导出 {layer_name} -> {stl_filename}")
            
            try:
                # 使用STLExporter导出单层，传递全局偏移量
                layer_options = options.copy()
                layer_options["single_layer_index"] = layer_idx
                if global_offset is not None:
                    layer_options["global_coord_offset"] = global_offset
                
                self.stl_exporter.export(data, stl_filepath, layer_options)
                
                # 记录文件信息
                file_size = os.path.getsize(stl_filepath)
                exported_files.append(stl_filename)
                
                manifest_data["layers"].append({
                    "index": layer_idx,
                    "name": layer_name,  # 中文名称
                    "name_english": english_name,  # 英文名称
                    "filename": stl_filename,
                    "file_size_bytes": file_size,
                    "file_size_mb": round(file_size / 1024 / 1024, 2)
                })
                
                print(f"    [OK] 成功 ({file_size / 1024:.1f} KB)")
                
            except Exception as e:
                print(f"    [ERROR] 失败: {e}")
                manifest_data["layers"].append({
                    "index": layer_idx,
                    "name": layer_name,
                    "name_english": english_name,
                    "filename": None,
                    "error": str(e)
                })
        
        if not exported_files:
            raise ValueError("没有成功导出任何地层")
        
        # 生成manifest.json
        manifest_path = os.path.join(temp_dir, "manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        # 生成README.txt
        readme_path = os.path.join(temp_dir, "README.txt")
        self._generate_readme(readme_path, manifest_data)
        
        # 生成FLAC3D导入脚本
        if options.get("include_fish_script", True):
            fish_script_path = os.path.join(temp_dir, "import_to_flac3d.fish")
            self._generate_fish_script(fish_script_path, manifest_data)
        
        # 打包为ZIP
        print(f"\n[Layered STL Export] 打包文件...")
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加所有STL文件
            for filename in exported_files:
                file_path = os.path.join(temp_dir, filename)
                zipf.write(file_path, filename)
            
            # 添加元数据文件
            zipf.write(manifest_path, "manifest.json")
            zipf.write(readme_path, "README.txt")
            
            if options.get("include_fish_script", True):
                zipf.write(fish_script_path, "import_to_flac3d.fish")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        
        zip_size = os.path.getsize(output_zip_path)
        print(f"[Layered STL Export] [DONE] 完成！")
        print(f"  - 导出地层: {len(exported_files)}/{len(layers)}")
        print(f"  - 文件大小: {zip_size / 1024 / 1024:.2f} MB")
        print(f"  - 保存位置: {output_zip_path}")
        
        return output_zip_path
    
    def _to_english_filename(self, name: str) -> str:
        """
        将中文岩层名转换为英文文件名
        优先使用预定义的地质专业术语对照表，否则使用拼音
        
        Args:
            name: 中文岩层名称（如：6煤、砂质泥岩）
        
        Returns:
            str: 英文文件名（如：coal_6、sandy_mudstone）
        """
        # 地质专业术语对照表
        geology_terms = {
            # 煤层
            '煤': 'coal',
            '煤层': 'coal_seam',
            
            # 岩石类型
            '泥岩': 'mudstone',
            '砂岩': 'sandstone',
            '页岩': 'shale',
            '石灰岩': 'limestone',
            '砂质泥岩': 'sandy_mudstone',
            '泥质砂岩': 'muddy_sandstone',
            '粉砂岩': 'siltstone',
            '砾岩': 'conglomerate',
            '花岗岩': 'granite',
            '玄武岩': 'basalt',
            '片岩': 'schist',
            '片麻岩': 'gneiss',
            '板岩': 'slate',
            '大理岩': 'marble',
            '石英岩': 'quartzite',
            
            # 矿物
            '石膏': 'gypsum',
            '盐岩': 'salt_rock',
            '硬石膏': 'anhydrite',
            
            # 土层
            '黏土': 'clay',
            '粉土': 'silt',
            '砂土': 'sand',
            '砾石': 'gravel',
            '黄土': 'loess',
            
            # 修饰词
            '细': 'fine',
            '中': 'medium',
            '粗': 'coarse',
            '薄': 'thin',
            '厚': 'thick',
            '坚硬': 'hard',
            '软': 'soft',
            '质': '',  # 砂质、泥质等的"质"字可以省略
        }
        
        # 首先尝试整体匹配
        if name in geology_terms:
            return geology_terms[name]
        
        # 提取数字和汉字部分
        result_parts = []
        
        # 处理带数字的煤层名（如：6煤、11煤）
        coal_pattern = r'(\d+)煤'
        coal_match = re.match(coal_pattern, name)
        if coal_match:
            number = coal_match.group(1)
            return f"coal_{number}"
        
        # 处理复合名称（如：砂质泥岩、细砂岩）
        temp_name = name
        matched_terms = []
        
        # 尝试匹配多个术语组合，保持顺序
        for cn_term in sorted(geology_terms.keys(), key=len, reverse=True):
            if cn_term in temp_name:
                en_term = geology_terms[cn_term]
                if en_term:  # 忽略空字符串（如"质"）
                    # 记录匹配位置和术语
                    pos = temp_name.find(cn_term)
                    matched_terms.append((pos, en_term))
                temp_name = temp_name.replace(cn_term, ' ', 1)  # 用空格替换而不是删除
        
        matched = len(matched_terms) > 0
        
        # 如果有匹配的术语
        if matched:
            # 按匹配位置排序，保持原始顺序
            matched_terms.sort(key=lambda x: x[0])
            result_parts = [term for pos, term in matched_terms]
            
            # 处理剩余的数字
            numbers = re.findall(r'\d+', temp_name)
            if numbers:
                result_parts.extend(numbers)
            
            result = '_'.join(result_parts)
            return self._sanitize_filename(result)
        
        # 如果没有匹配，使用拼音转换
        if HAS_PYPINYIN:
            # 分离数字和汉字
            parts = re.split(r'(\d+)', name)
            result_parts = []
            
            for part in parts:
                if part.isdigit():
                    result_parts.append(part)
                elif part.strip():
                    # 转换为拼音
                    pinyin_list = lazy_pinyin(part, style=Style.NORMAL)
                    result_parts.extend(pinyin_list)
            
            result = '_'.join(result_parts)
        else:
            # 如果没有pypinyin库，使用简单的编号
            result = f"layer_{name}"
        
        return self._sanitize_filename(result)
    
    def _sanitize_filename(self, name: str) -> str:
        """清理文件名，移除非法字符，只保留英文、数字、下划线"""
        # 只保留字母、数字、下划线和连字符
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        # 移除多余的下划线
        name = re.sub(r'_+', '_', name)
        # 移除首尾下划线
        name = name.strip('_')
        # 转换为小写
        name = name.lower()
        return name if name else 'unnamed_layer'
    
    def _generate_readme(self, filepath: str, manifest: Dict):
        """生成README导入说明"""
        content = f"""# 地质模型分层STL文件包

## 导出信息
- 导出时间: {manifest['export_time']}
- 地层总数: {manifest['total_layers']}
- 成功导出: {len([l for l in manifest['layers'] if l.get('filename')])}
- STL格式: {manifest['format'].upper()}
- 降采样倍数: {manifest['downsample_factor']}x
- 坐标归一化: {'是' if manifest['coordinate_normalized'] else '否'}

## 文件列表

"""
        
        for layer_info in manifest['layers']:
            if layer_info.get('filename'):
                content += f"- {layer_info['filename']}\n"
                content += f"  岩层（中文）: {layer_info['name']}\n"
                content += f"  岩层（英文）: {layer_info.get('name_english', 'N/A')}\n"
                content += f"  大小: {layer_info['file_size_mb']} MB\n\n"
        
        content += """
## FLAC3D导入步骤

### 方法1: 使用自动脚本（推荐）

1. 解压此ZIP文件到本地目录（确保所有文件在同一文件夹）
2. 在FLAC3D中运行:
   ```
   program call "import_to_flac3d.fish"
   ```
3. 脚本采用线性执行方式，逐层导入并生成网格
4. 完成后会自动执行 zone attach by-face 缝合所有网格
5. 最终模型保存为 'Mesh_Generated_XXLayers.sav'

**调试提示**：
- 如果某层导入失败，可以直接在脚本中注释掉该层继续执行
- 可以修改 mesh_size 函数中的返回值来调整网格密度
- 脚本无复杂函数嵌套，易于理解和修改

### 方法2: 手动逐层导入

1. 解压此ZIP文件
2. 在FLAC3D中逐个导入:
   ```
   ; 导入第1层（示例：6煤层）
   geometry import stl "01_coal_6.stl"
   zone generate from-geometry edge-length 10.0
   
   ; 导入第2层（示例：砂质泥岩）
   geometry import stl "02_sandy_mudstone.stl"
   zone generate from-geometry edge-length 10.0
   
   ; ... 依此类推
   ```
   
   注意：文件名为英文（FLAC3D对中文支持不好），具体文件名请参考上方"文件列表"

3. 为每层设置材料属性:
   ```
   zone cmodel assign elastic
   zone property bulk 5e9 shear 3e9 density 2500
   ```

### 方法3: 合并导入（不推荐）

如果需要合并所有地层为一个模型，可以依次导入所有STL文件。
但这可能导致内部分层面问题，不利于网格生成。

## 注意事项

1. **逐层导入**: 每层作为独立的几何体导入，避免内部分层面
2. **网格生成**: 建议每层单独生成网格，再设置材料属性
3. **坐标系统**: 如果启用了坐标归一化，所有坐标已转换为相对坐标
4. **文件顺序**: 文件名前缀数字表示地层顺序（从底到顶）

## 技术支持

如遇到问题，请查看 manifest.json 获取详细的导出信息。

---
Generated by Geological Modeling System
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_fish_script(self, filepath: str, manifest: Dict):
        """生成FLAC3D自动导入脚本（线性执行，易于调试）"""
        
        total_layers = len([l for l in manifest['layers'] if l.get('filename')])
        
        script = f"""; ==========================================
; FLAC3D {total_layers}层地质模型分步导入脚本
; 生成时间: {manifest['export_time']}
; 优势：线性执行，无复杂函数，易于调试
; ==========================================

; --- 1. 初始化 ---
model new
model deterministic on
model title "{total_layers}-Layer Geological Model"

; --- 2. 定义全局网格尺寸 ---
; 修改这个数字可以控制所有层的网格疏密
fish define mesh_size
    return 50.0   ; 建议范围: 20.0 - 100.0 (单位：米)
end

; --- 3. 层间间隙配置 ---
; 自动层间间隙: {manifest.get('min_layer_gap', 0.5)}m
; 坐标归一化: {'是' if manifest.get('coordinate_normalized') else '否'}
;
; 说明：层间间隙不会影响建模，反而有以下好处：
;   1. 避免浮点误差导致的几何冲突
;   2. 提高接触面网格质量
;   3. 便于建立zone attach或zone interface
;   4. 使用zone attach可以刚性连接各层（无变形）
;
; 建立层间连接的方法（在所有层导入后执行）：
;   ; 方法1：刚性连接（假设完整接触）
;   zone attach by-face range geometry 'geo_01' range geometry 'geo_02'
;   
;   ; 方法2：接触单元（考虑滑移/分离）
;   zone interface create by-face range z [间隙中点Z值]
;   zone interface property stiffness-normal=1e10 stiffness-shear=1e9

; ==========================================
;   开始分层导入 (Layer 01 - {total_layers:02d})
; ==========================================
; 
; 重要提示：
; 如果遇到 "A hard edge is cut by another hard edge" 错误，
; 说明相邻层几何体有重叠或网格不闭合（已自动修复）。解决方法：
; 
; 方法1：逐层调试（推荐）
;   - 先注释掉所有层，只导入第1层
;   - 确认成功后，逐层添加，找出问题层
;   - 对问题层单独处理
; 
; 方法2：增大网格尺寸
;   - 将 mesh_size 改为 100.0 或更大
;   - 粗网格更容易生成，适合初步测试
; 
; 方法3：删除几何集后重新导入
;   - 如果某层失败，使用: geometry delete set 'geo_XX'
;   - 然后重新导入该层

; 【关键】升级版ID范围追踪函数
fish define prepare_id_range
    global max_id = 0
    loop foreach z zone.list
        max_id = math.max(max_id, zone.id(z))
    end_loop
    global id_lower = max_id + 1
end

; ==========================================
;   开始导入 (Layer 01 - """ + str(total_layers) + """)
; ==========================================

"""
        
        # 检查是否有顶板层（第一层名称包含"顶板"）
        has_top_plate = False
        if manifest['layers'] and manifest['layers'][0].get('filename'):
            first_layer_name = manifest['layers'][0]['name']
            if '顶板' in first_layer_name or 'layer' in manifest['layers'][0].get('name_english', '').lower():
                has_top_plate = True
        
        # 如果有顶板层,添加特别说明
        if has_top_plate:
            last_layer_num = len(manifest['layers'])
            script += f"""; ==========================================
;   🛡️ 顶板层说明 (Layer {last_layer_num:02d})
; ==========================================
; 最后一层为自动生成的顶板层,具有以下特点:
;   • 底面:跟随最顶层地质体的曲面起伏
;   • 顶面:完全平坦的水平面
;   • 用途:便于施加上覆载荷和设置顶部边界条件
;
; 推荐设置:
;   1. 材料属性:与最顶层相同或稍硬(代表上覆岩层)
;   2. 边界条件:固定顶面(zone face apply velocity-z 0 range group 'L{last_layer_num:02d}_*' face top)
;   3. 载荷施加:在顶面施加均布载荷(zone face apply stress-zz [压力值] range group 'L{last_layer_num:02d}_*' face top)
; ==========================================

"""
        
        # 为每层生成导入命令
        for idx, layer_info in enumerate(manifest['layers']):
            if not layer_info.get('filename'):
                continue
            
            filename = layer_info['filename']
            layer_name_cn = layer_info['name']
            layer_name_en = layer_info.get('name_english', f"layer_{layer_info['index']+1}")
            layer_num = layer_info['index'] + 1
            group_name = f"L{layer_num:02d}_{layer_name_en}"
            geo_set_name = f"geo_{layer_num:02d}"
            
            # 为顶板层添加特殊标记
            if idx == 0 and has_top_plate:
                script += f"""; --- Layer {layer_num:02d}: {layer_name_cn} (顶板层 - 顶面平坦) ---
@prepare_id_range
geometry import '{filename}' set '{geo_set_name}'
geometry set '{geo_set_name}' triangulate
zone generate from-geometry set '{geo_set_name}' maximum-edge @mesh_size
zone group '{group_name}' range id @id_lower 100000000

"""
            else:
                script += f"""; --- Layer {layer_num:02d}: {layer_name_cn} ---
@prepare_id_range
geometry import '{filename}' set '{geo_set_name}'
geometry set '{geo_set_name}' triangulate
zone generate from-geometry set '{geo_set_name}' maximum-edge @mesh_size
zone group '{group_name}' range id @id_lower 100000000

"""
        
        # 根据是否有顶板添加不同的后续步骤
        if has_top_plate:
            script += """
; ==========================================
;   3. 建立层间连接（关键步骤）
; ==========================================
; 虽然各层有物理间隙，但需要建立力学连接
zone attach by-face

; ==========================================
;   4. 顶板层专用配置（推荐）
; ==========================================
; 以下为顶板层的典型应用示例，根据实际需求选择

; --- 4.1 施加上覆载荷（均布压力）---
; 模拟上覆岩层自重，假设埋深500m，岩石密度2500kg/m³
fish define apply_overburden_load
    ; 计算上覆压力: P = ρ × g × h
    local depth = 500.0        ; 埋深(m)
    local density = 2500.0     ; 密度(kg/m³)
    local gravity = 9.81       ; 重力加速度(m/s²)
    local pressure = density * gravity * depth  ; 压力(Pa)
    
    ; 在顶板顶面施加压力
    command
        zone face apply stress-zz [pressure] range group 'L01_*' face top
    end_command
    
    io.out('已在顶板顶面施加上覆载荷: ' + string(pressure/1e6) + ' MPa')
end
; 执行载荷施加（取消下一行注释）
; [@apply_overburden_load]

; --- 4.2 固定顶板顶面（边界条件）---
; 如果不需要施加载荷，而是固定顶面，使用以下命令
; zone face apply velocity-z 0 range group 'L01_*' face top

; --- 4.3 查看顶板层信息---
fish define show_top_plate_info
    ; 统计顶板zone数量
    local count = 0
    loop foreach z zone.list
        if string.find(zone.group(z), 'L01_') # 0 then
            count = count + 1
        end_if
    end_loop
    io.out('顶板层zone数量: ' + string(count))
end
[@show_top_plate_info]

; ==========================================
;   5. 检查并显示结果
; ==========================================
"""
        else:
            script += """
; ==========================================
;   3. 建立层间连接（关键步骤）
; ==========================================
; 虽然各层有物理间隙，但需要建立力学连接
; 选择以下方法之一：

; --- 方法A：刚性连接（推荐，假设完整接触）---
; 将所有相邻层的接触面粘合在一起
; 优点：简单、稳定，适合大多数情况
zone attach by-face

; --- 方法B：柔性接触（可选，适合软弱夹层）---
; 如果需要模拟层间滑移或分离，使用接触单元
; 注释掉上面的 zone attach，改用以下代码：
;
; fish define setup_interfaces
;   ; 为每对相邻层创建接触界面
;   zone interface create by-face
;   ; 设置接触刚度（根据实际地质条件调整）
;   zone interface property stiffness-normal=1e10 stiffness-shear=1e9
;   zone interface property friction=30.0 cohesion=0.5e6
; end
; [@setup_interfaces]

; ==========================================
;   4. 检查并显示结果
; ==========================================
"""
        
        script += """
; ==========================================
;   3. 建立层间连接（关键步骤）
; ==========================================
; 虽然各层有物理间隙，但需要建立力学连接
; 选择以下方法之一：

; --- 方法A：刚性连接（推荐，假设完整接触）---
; 将所有相邻层的接触面粘合在一起
; 优点：简单、稳定，适合大多数情况
zone attach by-face

; --- 方法B：柔性接触（可选，适合软弱夹层）---
; 如果需要模拟层间滑移或分离，使用接触单元
; 注释掉上面的 zone attach，改用以下代码：
;
; fish define setup_interfaces
;   ; 为每对相邻层创建接触界面
;   zone interface create by-face
;   ; 设置接触刚度（根据实际地质条件调整）
;   zone interface property stiffness-normal=1e10 stiffness-shear=1e9
;   zone interface property friction=30.0 cohesion=0.5e6
; end
; [@setup_interfaces]

; ==========================================
;   4. 检查并显示结果
; ==========================================
; 显示模型信息
model list information

; ==========================================
;   5. 保存模型
; ==========================================
model save 'Mesh_Generated_""" + str(total_layers) + """Layers.sav'

; 输出结果信息
model list information

; ==========================================
;   导入完成！
; ==========================================
; 下一步：
; 1. 为各层分配材料属性
;    例如: zone cmodel assign elastic range group 'L01_coal_6'
;         zone property bulk 5e9 shear 3e9 density 2500 range group 'L01_coal_6'
;
; 2. 设置边界条件
;    例如: zone face apply velocity-normal 0 range position-z 0
;
; 3. 运行模拟
;    例如: model solve
;
; 注意：
; - 如果导入过程中出错，可以注释掉已成功导入的层继续调试
; - mesh_size 值越小网格越密集，计算量越大
; - 确保所有STL文件与此脚本在同一目录下
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(script)
