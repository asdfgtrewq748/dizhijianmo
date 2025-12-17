"""
地质岩石材质包系统
提供专业的PBR材质、纹理贴图和材质导出功能
支持导出到Blender、Maya、3ds Max等建模软件
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from PIL import Image
import json


@dataclass
class MaterialProperties:
    """材质物理属性（PBR参数）"""
    name: str
    base_color: Tuple[int, int, int]  # RGB 0-255
    metallic: float  # 0-1
    roughness: float  # 0-1
    normal_strength: float  # 法线贴图强度
    ao_strength: float  # 环境光遮蔽强度
    displacement_scale: float  # 位移贴图强度

    # 纹理生成参数
    grain_size: int = 3  # 颗粒尺寸
    layering: int = 20  # 层理数量
    variation: float = 0.15  # 颜色变化


# ==================== 专业岩石材质库 ====================

ROCK_MATERIALS = {
    # 煤层 - 黑色系，高光泽
    '煤': MaterialProperties(
        name='coal',
        base_color=(26, 26, 26),
        metallic=0.1,
        roughness=0.6,
        normal_strength=0.5,
        ao_strength=0.8,
        displacement_scale=0.02,
        layering=30,
        variation=0.1
    ),

    # 砂岩 - 黄褐色，粗糙颗粒
    '砂岩': MaterialProperties(
        name='sandstone',
        base_color=(196, 163, 90),
        metallic=0.0,
        roughness=0.85,
        normal_strength=0.8,
        ao_strength=0.6,
        displacement_scale=0.05,
        grain_size=3,
        variation=0.2
    ),

    '细砂岩': MaterialProperties(
        name='fine_sandstone',
        base_color=(212, 184, 120),
        metallic=0.0,
        roughness=0.75,
        normal_strength=0.6,
        ao_strength=0.5,
        displacement_scale=0.03,
        grain_size=2,
        variation=0.15
    ),

    '中砂岩': MaterialProperties(
        name='medium_sandstone',
        base_color=(196, 163, 90),
        metallic=0.0,
        roughness=0.8,
        normal_strength=0.75,
        ao_strength=0.55,
        displacement_scale=0.04,
        grain_size=3,
        variation=0.18
    ),

    '粗砂岩': MaterialProperties(
        name='coarse_sandstone',
        base_color=(180, 147, 71),
        metallic=0.0,
        roughness=0.9,
        normal_strength=0.9,
        ao_strength=0.65,
        displacement_scale=0.06,
        grain_size=5,
        variation=0.25
    ),

    # 泥岩 - 灰绿色，平滑
    '泥岩': MaterialProperties(
        name='mudstone',
        base_color=(107, 123, 107),
        metallic=0.0,
        roughness=0.7,
        normal_strength=0.4,
        ao_strength=0.5,
        displacement_scale=0.02,
        layering=15,
        variation=0.12
    ),

    '砂质泥岩': MaterialProperties(
        name='sandy_mudstone',
        base_color=(122, 138, 114),
        metallic=0.0,
        roughness=0.75,
        normal_strength=0.5,
        ao_strength=0.55,
        displacement_scale=0.03,
        grain_size=2,
        layering=10,
        variation=0.15
    ),

    # 页岩 - 深灰色，薄层状
    '页岩': MaterialProperties(
        name='shale',
        base_color=(74, 85, 104),
        metallic=0.05,
        roughness=0.5,
        normal_strength=0.7,
        ao_strength=0.7,
        displacement_scale=0.03,
        layering=50,
        variation=0.1
    ),

    '炭质页岩': MaterialProperties(
        name='carbonaceous_shale',
        base_color=(61, 72, 82),
        metallic=0.08,
        roughness=0.45,
        normal_strength=0.8,
        ao_strength=0.8,
        displacement_scale=0.025,
        layering=60,
        variation=0.08
    ),

    # 粉砂岩
    '粉砂岩': MaterialProperties(
        name='siltstone',
        base_color=(168, 144, 120),
        metallic=0.0,
        roughness=0.8,
        normal_strength=0.5,
        ao_strength=0.5,
        displacement_scale=0.03,
        grain_size=2,
        variation=0.15
    ),

    # 灰岩 - 灰蓝色，块状
    '灰岩': MaterialProperties(
        name='limestone',
        base_color=(139, 157, 170),
        metallic=0.1,
        roughness=0.4,
        normal_strength=0.6,
        ao_strength=0.6,
        displacement_scale=0.04,
        grain_size=4,
        variation=0.18
    ),

    # 砾岩 - 棕红色，大颗粒
    '砾岩': MaterialProperties(
        name='conglomerate',
        base_color=(139, 90, 60),
        metallic=0.0,
        roughness=0.95,
        normal_strength=1.0,
        ao_strength=0.7,
        displacement_scale=0.08,
        grain_size=10,
        variation=0.3
    ),

    # 表土/黏土
    '表土': MaterialProperties(
        name='soil',
        base_color=(181, 149, 108),
        metallic=0.0,
        roughness=0.95,
        normal_strength=0.4,
        ao_strength=0.5,
        displacement_scale=0.04,
        grain_size=2,
        variation=0.2
    ),

    '黏土': MaterialProperties(
        name='clay',
        base_color=(154, 128, 96),
        metallic=0.0,
        roughness=0.9,
        normal_strength=0.3,
        ao_strength=0.5,
        displacement_scale=0.02,
        variation=0.15
    ),
}


# ==================== 纹理生成器 ====================

class TextureMapGenerator:
    """生成完整的PBR纹理贴图集"""

    def __init__(self, resolution: int = 1024):
        """
        Args:
            resolution: 纹理分辨率（建议1024或2048）
        """
        self.resolution = resolution

    def generate_all_maps(
        self,
        material: MaterialProperties,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        生成完整的PBR纹理贴图集

        Returns:
            包含以下键的字典：
            - 'albedo': 基础颜色贴图 (RGB)
            - 'normal': 法线贴图 (RGB)
            - 'roughness': 粗糙度贴图 (灰度)
            - 'metallic': 金属度贴图 (灰度)
            - 'ao': 环境光遮蔽贴图 (灰度)
            - 'height': 高度/位移贴图 (灰度)
        """
        np.random.seed(seed)

        maps = {}

        # 1. 生成基础噪声
        base_noise = self._generate_base_noise(material)

        # 2. Albedo（基础颜色）贴图
        maps['albedo'] = self._generate_albedo_map(material, base_noise)

        # 3. Height（高度）贴图
        maps['height'] = self._generate_height_map(material, base_noise)

        # 4. Normal（法线）贴图 - 从高度图生成
        maps['normal'] = self._height_to_normal(maps['height'], material.normal_strength)

        # 5. Roughness（粗糙度）贴图
        maps['roughness'] = self._generate_roughness_map(material, base_noise)

        # 6. Metallic（金属度）贴图
        maps['metallic'] = self._generate_metallic_map(material)

        # 7. AO（环境光遮蔽）贴图
        maps['ao'] = self._generate_ao_map(material, maps['height'])

        return maps

    def _generate_base_noise(self, material: MaterialProperties) -> np.ndarray:
        """生成基础噪声纹理"""
        size = self.resolution

        # 根据岩石类型生成不同的噪声模式
        if 'coal' in material.name or 'shale' in material.name:
            # 层状纹理
            noise = self._layered_noise(size, material.layering)
        elif 'sand' in material.name:
            # 颗粒状纹理
            noise = self._granular_noise(size, material.grain_size)
        elif 'mud' in material.name or 'clay' in material.name or 'silt' in material.name:
            # 平滑层理
            noise = self._smooth_layered_noise(size)
        elif 'limestone' in material.name:
            # 块状纹理
            noise = self._blocky_noise(size)
        elif 'conglomerate' in material.name:
            # 大颗粒纹理
            noise = self._coarse_granular_noise(size, material.grain_size)
        else:
            # 默认混合纹理
            noise = self._perlin_like_noise(size)

        return noise

    def _layered_noise(self, size: int, layers: int) -> np.ndarray:
        """层状噪声（煤层、页岩）"""
        y = np.linspace(0, layers * 2 * np.pi, size)
        base = np.sin(y)[:, np.newaxis]
        base = np.tile(base, (1, size))

        # 添加随机扰动
        noise = np.random.randn(size, size) * 0.3
        noise = gaussian_filter(noise, sigma=2)

        result = (base + noise) * 0.5 + 0.5
        return np.clip(result, 0, 1)

    def _granular_noise(self, size: int, grain_size: int) -> np.ndarray:
        """颗粒状噪声（砂岩）"""
        low_size = max(2, size // grain_size)
        noise = np.random.rand(low_size, low_size)

        # 上采样
        from scipy.ndimage import zoom
        upsampled = zoom(noise, size / low_size, order=1)

        # 添加细节
        detail = np.random.randn(size, size) * 0.2
        detail = gaussian_filter(detail, sigma=1)

        result = upsampled + detail
        return np.clip(result, 0, 1)

    def _smooth_layered_noise(self, size: int) -> np.ndarray:
        """平滑层理噪声（泥岩）"""
        y = np.linspace(0, 4 * np.pi, size)
        x = np.linspace(0, 8 * np.pi, size)
        X, Y = np.meshgrid(x, y)

        base = np.sin(Y) * 0.3 + np.sin(X * 0.2) * 0.2 + 0.5
        noise = np.random.randn(size, size) * 0.1
        noise = gaussian_filter(noise, sigma=3)

        result = base + noise
        return np.clip(result, 0, 1)

    def _blocky_noise(self, size: int) -> np.ndarray:
        """块状噪声（灰岩）"""
        # Voronoi风格
        n_points = 100
        points_y = np.random.randint(0, size, n_points)
        points_x = np.random.randint(0, size, n_points)
        values = np.random.rand(n_points)

        Y, X = np.mgrid[0:size, 0:size]
        noise = np.zeros((size, size))

        for i in range(n_points):
            dist = np.sqrt((Y - points_y[i])**2 + (X - points_x[i])**2)
            mask = dist < size / 10
            if mask.any():
                noise[mask] = values[i]

        noise = gaussian_filter(noise, sigma=8)
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    def _coarse_granular_noise(self, size: int, grain_size: int) -> np.ndarray:
        """粗颗粒噪声（砾岩）"""
        low_size = max(2, size // grain_size)
        noise = np.random.rand(low_size, low_size)

        from scipy.ndimage import zoom
        upsampled = zoom(noise, size / low_size, order=0)  # 最近邻，保持块状
        upsampled = gaussian_filter(upsampled, sigma=2)

        return upsampled

    def _perlin_like_noise(self, size: int) -> np.ndarray:
        """Perlin风格噪声"""
        noise = np.zeros((size, size))

        for octave in range(5):
            freq = 2 ** octave
            amp = 0.5 ** octave

            low_size = max(2, size // (10 * freq))
            low_noise = np.random.randn(low_size, low_size)

            from scipy.ndimage import zoom
            upsampled = zoom(low_noise, size / low_size, order=1)
            noise += upsampled * amp

        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise

    def _generate_albedo_map(
        self,
        material: MaterialProperties,
        base_noise: np.ndarray
    ) -> np.ndarray:
        """生成Albedo（基础颜色）贴图"""
        base_color = np.array(material.base_color) / 255.0

        # 颜色变化
        color_variation = (base_noise * 2 - 1) * material.variation

        # 应用到每个通道
        albedo = np.zeros((*base_noise.shape, 3))
        for c in range(3):
            albedo[:, :, c] = base_color[c] * (1 + color_variation)

        albedo = np.clip(albedo, 0, 1)
        return (albedo * 255).astype(np.uint8)

    def _generate_height_map(
        self,
        material: MaterialProperties,
        base_noise: np.ndarray
    ) -> np.ndarray:
        """生成Height（高度）贴图"""
        # 高度贴图就是基础噪声，但可以调整对比度
        height = base_noise * material.displacement_scale
        return (height * 255).astype(np.uint8)

    def _height_to_normal(self, height_map: np.ndarray, strength: float) -> np.ndarray:
        """从高度图生成法线贴图"""
        height = height_map.astype(float) / 255.0

        # Sobel算子计算梯度
        dy, dx = np.gradient(height)

        # 法向量
        normal = np.zeros((*height.shape, 3))
        normal[:, :, 0] = -dx * strength  # R通道 (X)
        normal[:, :, 1] = -dy * strength  # G通道 (Y)
        normal[:, :, 2] = 1.0              # B通道 (Z)

        # 归一化
        magnitude = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = normal / (magnitude + 1e-8)

        # 映射到0-255范围 (OpenGL标准)
        normal = (normal * 0.5 + 0.5) * 255
        return normal.astype(np.uint8)

    def _generate_roughness_map(
        self,
        material: MaterialProperties,
        base_noise: np.ndarray
    ) -> np.ndarray:
        """生成Roughness（粗糙度）贴图"""
        # 基础粗糙度 + 噪声变化
        roughness = material.roughness + (base_noise - 0.5) * 0.2
        roughness = np.clip(roughness, 0, 1)
        return (roughness * 255).astype(np.uint8)

    def _generate_metallic_map(self, material: MaterialProperties) -> np.ndarray:
        """生成Metallic（金属度）贴图"""
        # 大部分岩石金属度是均匀的
        metallic = np.full(
            (self.resolution, self.resolution),
            material.metallic * 255,
            dtype=np.uint8
        )
        return metallic

    def _generate_ao_map(
        self,
        material: MaterialProperties,
        height_map: np.ndarray
    ) -> np.ndarray:
        """生成AO（环境光遮蔽）贴图"""
        # 基于高度图生成简单的AO
        height = height_map.astype(float) / 255.0

        # 模糊高度图，低处较暗
        blurred = gaussian_filter(height, sigma=5)
        ao = 1.0 - (1.0 - blurred) * material.ao_strength

        return (ao * 255).astype(np.uint8)


# ==================== 材质导出器 ====================

class MaterialExporter:
    """导出材质到各种3D软件格式"""

    @staticmethod
    def export_material_pack(
        rock_name: str,
        output_dir: str,
        resolution: int = 1024,
        formats: List[str] = ['blender', 'unity', 'unreal']
    ) -> Dict[str, str]:
        """
        导出完整的材质包

        Args:
            rock_name: 岩石名称（如'砂岩'）
            output_dir: 输出目录
            resolution: 纹理分辨率
            formats: 导出格式列表

        Returns:
            导出的文件路径字典
        """
        # 获取材质属性
        material = MaterialExporter._get_material(rock_name)
        if material is None:
            raise ValueError(f"未找到岩石材质: {rock_name}")

        # 创建输出目录
        mat_dir = Path(output_dir) / material.name
        mat_dir.mkdir(parents=True, exist_ok=True)

        # 生成纹理贴图
        print(f"正在生成 {rock_name} 的纹理贴图...")
        generator = TextureMapGenerator(resolution=resolution)
        texture_maps = generator.generate_all_maps(material)

        # 保存纹理文件
        texture_files = {}
        for map_name, map_data in texture_maps.items():
            filename = f"{material.name}_{map_name}.png"
            filepath = mat_dir / filename

            if map_data.ndim == 2:
                # 灰度图
                Image.fromarray(map_data, mode='L').save(filepath)
            else:
                # RGB图
                Image.fromarray(map_data, mode='RGB').save(filepath)

            texture_files[map_name] = str(filepath)
            print(f"  - 已保存: {filename}")

        # 导出材质定义文件
        exported_files = {'textures': texture_files}

        if 'blender' in formats:
            exported_files['blender'] = MaterialExporter._export_blender_material(
                material, texture_files, mat_dir
            )

        if 'unity' in formats:
            exported_files['unity'] = MaterialExporter._export_unity_material(
                material, texture_files, mat_dir
            )

        if 'unreal' in formats:
            exported_files['unreal'] = MaterialExporter._export_unreal_material(
                material, texture_files, mat_dir
            )

        # 保存材质配置JSON
        config_file = mat_dir / f"{material.name}_config.json"
        config = {
            'name': rock_name,
            'material_name': material.name,
            'base_color': material.base_color,
            'pbr_properties': {
                'metallic': material.metallic,
                'roughness': material.roughness,
                'normal_strength': material.normal_strength,
                'ao_strength': material.ao_strength,
                'displacement_scale': material.displacement_scale
            },
            'texture_files': {k: Path(v).name for k, v in texture_files.items()},
            'resolution': resolution
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        exported_files['config'] = str(config_file)

        print(f"\n✅ 材质包导出完成: {mat_dir}")
        return exported_files

    @staticmethod
    def _get_material(rock_name: str) -> Optional[MaterialProperties]:
        """获取岩石材质"""
        # 精确匹配
        if rock_name in ROCK_MATERIALS:
            return ROCK_MATERIALS[rock_name]

        # 模糊匹配
        for key, mat in ROCK_MATERIALS.items():
            if key in rock_name or rock_name in key:
                return mat

        return None

    @staticmethod
    def _export_blender_material(
        material: MaterialProperties,
        texture_files: Dict[str, str],
        output_dir: Path
    ) -> str:
        """导出Blender材质脚本"""
        script = f'''# Blender材质脚本: {material.name}
# 在Blender的Scripting标签页中运行此脚本

import bpy
import os

# 材质名称
mat_name = "{material.name}"

# 创建新材质
mat = bpy.data.materials.new(name=mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# 清除默认节点
nodes.clear()

# 纹理文件路径（相对于此脚本）
texture_dir = r"{output_dir}"

# 创建节点
output = nodes.new('ShaderNodeOutputMaterial')
output.location = (600, 0)

bsdf = nodes.new('ShaderNodeBsdfPrincipled')
bsdf.location = (300, 0)

# 基础颜色（Albedo）
albedo_tex = nodes.new('ShaderNodeTexImage')
albedo_tex.location = (-300, 300)
albedo_tex.image = bpy.data.images.load(os.path.join(texture_dir, "{Path(texture_files['albedo']).name}"))
links.new(albedo_tex.outputs['Color'], bsdf.inputs['Base Color'])

# 粗糙度
roughness_tex = nodes.new('ShaderNodeTexImage')
roughness_tex.location = (-300, 0)
roughness_tex.image = bpy.data.images.load(os.path.join(texture_dir, "{Path(texture_files['roughness']).name}"))
roughness_tex.image.colorspace_settings.name = 'Non-Color'
links.new(roughness_tex.outputs['Color'], bsdf.inputs['Roughness'])

# 金属度
metallic_tex = nodes.new('ShaderNodeTexImage')
metallic_tex.location = (-300, -300)
metallic_tex.image = bpy.data.images.load(os.path.join(texture_dir, "{Path(texture_files['metallic']).name}"))
metallic_tex.image.colorspace_settings.name = 'Non-Color'
links.new(metallic_tex.outputs['Color'], bsdf.inputs['Metallic'])

# 法线贴图
normal_tex = nodes.new('ShaderNodeTexImage')
normal_tex.location = (-600, -600)
normal_tex.image = bpy.data.images.load(os.path.join(texture_dir, "{Path(texture_files['normal']).name}"))
normal_tex.image.colorspace_settings.name = 'Non-Color'

normal_map = nodes.new('ShaderNodeNormalMap')
normal_map.location = (-300, -600)
normal_map.inputs['Strength'].default_value = {material.normal_strength}
links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

# 位移贴图（Height）
disp_tex = nodes.new('ShaderNodeTexImage')
disp_tex.location = (-300, -900)
disp_tex.image = bpy.data.images.load(os.path.join(texture_dir, "{Path(texture_files['height']).name}"))
disp_tex.image.colorspace_settings.name = 'Non-Color'

disp_node = nodes.new('ShaderNodeDisplacement')
disp_node.location = (300, -900)
disp_node.inputs['Scale'].default_value = {material.displacement_scale}
links.new(disp_tex.outputs['Color'], disp_node.inputs['Height'])
links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

# 连接BSDF到输出
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

print(f"✅ 材质 '{{mat_name}}' 创建完成!")
'''

        script_file = output_dir / f"{material.name}_blender.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script)

        return str(script_file)

    @staticmethod
    def _export_unity_material(
        material: MaterialProperties,
        texture_files: Dict[str, str],
        output_dir: Path
    ) -> str:
        """导出Unity材质配置"""
        # Unity使用.mat文件（YAML格式）
        # 这里生成一个说明文档
        doc = f'''# Unity材质设置: {material.name}

## 导入纹理
1. 将所有PNG文件拖入Unity的Assets文件夹
2. 设置纹理类型:
   - {Path(texture_files['albedo']).name}: Texture Type = Default (sRGB)
   - {Path(texture_files['normal']).name}: Texture Type = Normal Map
   - {Path(texture_files['roughness']).name}: Texture Type = Default (Linear)
   - {Path(texture_files['metallic']).name}: Texture Type = Default (Linear)
   - {Path(texture_files['height']).name}: Texture Type = Default (Linear)
   - {Path(texture_files['ao']).name}: Texture Type = Default (Linear)

## 创建材质
1. 右键 Assets -> Create -> Material
2. 设置材质为 Standard (Specular setup) 或 URP/Lit
3. 按如下设置纹理:

### Albedo (Base Map)
- Texture: {Path(texture_files['albedo']).name}

### Normal Map
- Texture: {Path(texture_files['normal']).name}
- Normal Strength: {material.normal_strength}

### Roughness (Smoothness)
- Texture: {Path(texture_files['roughness']).name}
- Note: Unity使用Smoothness，需要反转粗糙度贴图

### Metallic
- Texture: {Path(texture_files['metallic']).name}
- Metallic: {material.metallic}

### Height Map (Displacement)
- Texture: {Path(texture_files['height']).name}
- Height Scale: {material.displacement_scale}

### Ambient Occlusion
- Texture: {Path(texture_files['ao']).name}
'''

        doc_file = output_dir / f"{material.name}_unity_guide.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc)

        return str(doc_file)

    @staticmethod
    def _export_unreal_material(
        material: MaterialProperties,
        texture_files: Dict[str, str],
        output_dir: Path
    ) -> str:
        """导出Unreal Engine材质配置"""
        doc = f'''# Unreal Engine材质设置: {material.name}

## 导入纹理
1. 将所有PNG文件拖入Content Browser
2. 设置纹理压缩类型:
   - {Path(texture_files['albedo']).name}: Compression = Default (sRGB)
   - {Path(texture_files['normal']).name}: Compression = Normalmap
   - {Path(texture_files['roughness']).name}: Compression = Masks (Linear)
   - {Path(texture_files['metallic']).name}: Compression = Masks (Linear)
   - {Path(texture_files['height']).name}: Compression = Masks (Linear)
   - {Path(texture_files['ao']).name}: Compression = Masks (Linear)

## 创建材质
1. 右键 Content Browser -> Material
2. 命名为 M_{material.name}
3. 双击打开材质编辑器

## 节点连接
在材质编辑器中创建以下节点并连接:

### Base Color
- Texture Sample -> {Path(texture_files['albedo']).name}
- 连接到 Material 的 Base Color 输入

### Normal
- Texture Sample -> {Path(texture_files['normal']).name}
- 连接到 Material 的 Normal 输入
- 注意: 纹理属性设置为 Normal

### Roughness
- Texture Sample -> {Path(texture_files['roughness']).name}
- 连接到 Material 的 Roughness 输入

### Metallic
- Texture Sample -> {Path(texture_files['metallic']).name}
- Multiply by Constant ({material.metallic})
- 连接到 Material 的 Metallic 输入

### Ambient Occlusion
- Texture Sample -> {Path(texture_files['ao']).name}
- 连接到 Material 的 Ambient Occlusion 输入

### Displacement (可选)
- Texture Sample -> {Path(texture_files['height']).name}
- Multiply by {material.displacement_scale}
- 需要在材质属性中启用 Tessellation

## PBR参数
- Material Domain: Surface
- Blend Mode: Opaque
- Shading Model: Default Lit
'''

        doc_file = output_dir / f"{material.name}_unreal_guide.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc)

        return str(doc_file)


# ==================== 便捷函数 ====================

def export_all_rock_materials(
    output_dir: str = 'output/materials',
    resolution: int = 1024,
    formats: List[str] = ['blender', 'unity', 'unreal']
):
    """
    导出所有岩石材质包

    Args:
        output_dir: 输出根目录
        resolution: 纹理分辨率
        formats: 导出格式
    """
    print(f"开始导出所有岩石材质包...")
    print(f"输出目录: {output_dir}")
    print(f"纹理分辨率: {resolution}x{resolution}")
    print(f"导出格式: {', '.join(formats)}")
    print("-" * 60)

    exporter = MaterialExporter()
    results = {}

    for rock_name in ROCK_MATERIALS.keys():
        try:
            print(f"\n正在导出: {rock_name}")
            files = exporter.export_material_pack(
                rock_name,
                output_dir,
                resolution=resolution,
                formats=formats
            )
            results[rock_name] = files
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"✅ 完成! 共导出 {len(results)} 个材质包")
    print(f"输出目录: {Path(output_dir).absolute()}")

    return results


def get_material_preview(rock_name: str, size: int = 512) -> np.ndarray:
    """
    生成材质预览图

    Args:
        rock_name: 岩石名称
        size: 预览图尺寸

    Returns:
        RGB图像数组
    """
    material = MaterialExporter._get_material(rock_name)
    if material is None:
        raise ValueError(f"未找到材质: {rock_name}")

    generator = TextureMapGenerator(resolution=size)
    maps = generator.generate_all_maps(material)

    return maps['albedo']


if __name__ == "__main__":
    # 测试：导出单个材质
    print("测试材质导出系统...")

    # 导出砂岩材质
    files = MaterialExporter.export_material_pack(
        '砂岩',
        output_dir='output/materials_test',
        resolution=1024,
        formats=['blender', 'unity', 'unreal']
    )

    print("\n导出的文件:")
    for key, value in files.items():
        print(f"{key}: {value}")
