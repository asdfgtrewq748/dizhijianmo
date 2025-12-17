"""
FLAC3D 增强导出器测试脚本

测试功能:
1. 层间节点共享验证
2. FLAC3D命令脚本生成
3. 网格质量检查
4. 完整分析脚本生成
"""

import numpy as np
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from exporters.flac3d_enhanced_exporter import EnhancedFLAC3DExporter


def create_test_geological_model():
    """创建测试用地质模型 - 3层地质体"""
    print('='*60)
    print('  创建测试地质模型')
    print('='*60)

    # 定义网格
    nx, ny = 10, 10
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    XI, YI = np.meshgrid(x, y)

    # 定义3层地质体（从下到上）
    layers = []

    # 第1层 - 泥岩（底层）
    base_z = 0.0
    thickness1 = 10.0 + 2 * np.sin(XI/20) * np.cos(YI/20)  # 起伏的厚度
    top_z1 = base_z + thickness1

    layers.append({
        'name': '泥岩',
        'grid_x': x,  # 1D数组
        'grid_y': y,  # 1D数组
        'top_surface_z': top_z1,
        'bottom_surface_z': np.full_like(XI, base_z),
        'properties': {
            'density': 2400,
            'youngs_modulus': 5e9,
            'poisson_ratio': 0.25,
            'cohesion': 2e6,
            'friction_angle': 30
        }
    })

    # 第2层 - 煤层（中层）
    thickness2 = 3.0 + 0.5 * np.sin(XI/15) * np.cos(YI/15)
    top_z2 = top_z1 + thickness2

    layers.append({
        'name': '煤层',
        'grid_x': x,
        'grid_y': y,
        'top_surface_z': top_z2,
        'bottom_surface_z': top_z1.copy(),  # 关键：底面=下层顶面
        'properties': {
            'density': 1400,
            'youngs_modulus': 2e9,
            'poisson_ratio': 0.3,
            'cohesion': 1e6,
            'friction_angle': 25
        }
    })

    # 第3层 - 砂岩（顶层）
    thickness3 = 8.0 + 1.5 * np.sin(XI/25) * np.cos(YI/25)
    top_z3 = top_z2 + thickness3

    layers.append({
        'name': '砂岩',
        'grid_x': x,
        'grid_y': y,
        'top_surface_z': top_z3,
        'bottom_surface_z': top_z2.copy(),  # 关键：底面=下层顶面
        'properties': {
            'density': 2600,
            'youngs_modulus': 15e9,
            'poisson_ratio': 0.2,
            'cohesion': 5e6,
            'friction_angle': 35
        }
    })

    print(f'[OK] 创建了 {len(layers)} 层地质模型')
    for i, layer in enumerate(layers):
        print(f'     第{i+1}层: {layer["name"]}')
        print(f'           平均厚度: {np.mean(layer["top_surface_z"] - layer["bottom_surface_z"]):.2f} m')

    return layers, XI, YI


def test_flac3d_export():
    """测试FLAC3D增强导出器"""

    # 1. 创建测试模型
    layers, XI, YI = create_test_geological_model()

    # 2. 创建导出器
    print()
    print('='*60)
    print('  初始化FLAC3D增强导出器')
    print('='*60)

    exporter = EnhancedFLAC3DExporter()

    print('[OK] 导出器初始化完成')

    # 3. 执行导出
    print()
    print('='*60)
    print('  导出FLAC3D模型')
    print('='*60)

    output_dir = Path('output/flac3d_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'geological_model.f3dat'

    # 准备数据格式
    export_data = {
        'layers': layers,
        'title': '三层地质模型测试',
        'author': 'GNN地质建模系统'
    }

    # 导出选项
    export_options = {
        'normalize_coords': False,  # 使用原始坐标
        'validate_mesh': True,
        'coord_precision': 3
    }

    exporter.export(
        data=export_data,
        output_path=str(output_file),
        options=export_options
    )

    print(f'[OK] 模型已导出到: {output_file}')
    print(f'     文件大小: {output_file.stat().st_size / 1024:.2f} KB')

    # 4. 显示统计信息
    print()
    print('='*60)
    print('  导出统计')
    print('='*60)
    stats = exporter.stats

    print(f'总节点数: {stats["total_nodes"]}')
    print(f'共享节点数: {stats["shared_nodes"]}')
    print(f'总单元数: {stats["total_zones"]}')
    print(f'层数: {len(layers)}')
    if stats["total_zones"] > 0:
        print(f'平均每单元节点引用数: {(stats["total_zones"] * 8) / max(stats["total_nodes"], 1):.2f}')
    print()

    if stats.get('negative_volume_zones', 0) > 0:
        print(f'[WARNING] 负体积单元数: {stats["negative_volume_zones"]} (已自动修复)')
    else:
        print(f'[OK] 网格质量良好，无负体积单元')

    # 5. 生成完整分析脚本
    print()
    print('='*60)
    print('  生成FLAC3D分析脚本')
    print('='*60)

    analysis_script = output_dir / 'run_analysis.f3dat'
    with open(analysis_script, 'w', encoding='utf-8') as f:
        f.write('; FLAC3D 完整分析脚本\n')
        f.write('; 自动生成 - GNN地质建模系统\n\n')

        f.write('; 1. 导入模型\n')
        f.write(f"program call '{output_file.name}'\n\n")

        f.write('; 2. 设置本构模型\n')
        f.write('zone cmodel assign mohr-coulomb\n\n')

        f.write('; 3. 设置材料属性\n')
        for layer in layers:
            props = layer['properties']
            f.write(f"; {layer['name']}\n")
            f.write(f"zone property density={props['density']} shear={props['youngs_modulus']/(2*(1+props['poisson_ratio'])):.2e} bulk={props['youngs_modulus']/(3*(1-2*props['poisson_ratio'])):.2e} range group '{layer['name']}'\n")
            f.write(f"zone property cohesion={props['cohesion']:.2e} friction={props['friction_angle']} range group '{layer['name']}'\n\n")

        f.write('; 4. 边界条件\n')
        f.write('zone face apply velocity-x 0 range position-x 0\n')
        f.write('zone face apply velocity-x 0 range position-x 100\n')
        f.write('zone face apply velocity-y 0 range position-y 0\n')
        f.write('zone face apply velocity-y 0 range position-y 100\n')
        f.write('zone face apply velocity-z 0 range position-z 0\n\n')

        f.write('; 5. 重力初始化\n')
        f.write('model gravity 0 0 -9.81\n')
        f.write('zone initialize-stresses ratio 0.5\n\n')

        f.write('; 6. 求解到平衡\n')
        f.write('model solve ratio-average 1e-5\n')
        f.write('model save "initial_equilibrium.f3sav"\n\n')

        f.write('; 完成\n')
        f.write('program log-file off\n')

    print(f'[OK] 分析脚本已生成: {analysis_script.name}')

    # 6. 生成网格质量检查脚本
    quality_check_script = output_dir / 'check_mesh_quality.f3dat'
    with open(quality_check_script, 'w', encoding='utf-8') as f:
        f.write('; FLAC3D 网格质量检查脚本\n\n')
        f.write(f"program call '{output_file.name}'\n\n")
        f.write('; 检查单元质量\n')
        f.write('zone list quality\n\n')
        f.write('; 检查节点共享情况\n')
        f.write('zone list information\n')

    print(f'[OK] 质量检查脚本已生成: {quality_check_script.name}')

    # 7. 使用说明
    print()
    print('='*60)
    print('  在FLAC3D中使用')
    print('='*60)
    print()
    print('1. 导入模型:')
    print(f'   program call "{output_file.name}"')
    print()
    print('2. 运行完整分析:')
    print(f'   program call "{analysis_script.name}"')
    print()
    print('3. 检查网格质量:')
    print(f'   program call "{quality_check_script.name}"')
    print()
    print('='*60)
    print('  测试完成!')
    print('='*60)


if __name__ == '__main__':
    test_flac3d_export()
