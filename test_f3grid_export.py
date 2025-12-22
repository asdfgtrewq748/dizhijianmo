"""
FLAC3D f3grid 导出器 - 完整测试示例

演示如何使用所有优化功能：
1. B8 节点顺序修正
2. 薄层自动合并
3. 层间强制贴合
4. 角点厚度检查
5. 四面体体积检查
"""

import numpy as np
from src.exporters.f3grid_exporter_v2 import F3GridExporterV2


def create_test_model():
    """创建测试地质模型"""

    # 网格参数
    nx, ny = 20, 20
    x = np.linspace(0, 200, nx)
    y = np.linspace(0, 200, ny)
    XI, YI = np.meshgrid(x, y)

    # 基准高程（带点起伏）
    base_z = -100 + 0.1 * XI + 0.05 * YI

    # 创建多层地质模型（包含薄层）
    layers = [
        # 底板
        {
            'name': '底板泥岩',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z,
            'top_surface_z': base_z + 5.0,
        },
        # 薄煤层 1（会被合并）
        {
            'name': '16-6煤',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 5.0,
            'top_surface_z': base_z + 5.3,  # 仅 0.3m
        },
        # 薄煤层 2（会被合并）
        {
            'name': '16-5煤',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 5.3,
            'top_surface_z': base_z + 5.6,  # 仅 0.3m
        },
        # 薄煤层 3（会被合并）
        {
            'name': '16-4煤',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 5.6,
            'top_surface_z': base_z + 6.1,  # 0.5m
        },
        # 中间层
        {
            'name': '夹矸泥岩',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 6.1,
            'top_surface_z': base_z + 8.0,
        },
        # 主煤层（较厚，不会被合并）
        {
            'name': '16-3煤',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 8.0,
            'top_surface_z': base_z + 10.5,  # 2.5m
        },
        # 顶板
        {
            'name': '顶板砂岩',
            'grid_x': x,
            'grid_y': y,
            'bottom_surface_z': base_z + 10.5,
            'top_surface_z': base_z + 20.0,
        },
    ]

    return {'layers': layers}


def test_basic_export():
    """测试1: 基本导出（不使用优化功能）"""
    print("=" * 70)
    print("测试1: 基本导出（不使用优化功能）")
    print("=" * 70)

    data = create_test_model()
    exporter = F3GridExporterV2()

    options = {
        'downsample_factor': 1,
        'uniform_downsample': True,
        'merge_thin_layers': False,        # 不合并
        'force_layer_continuity': False,   # 不强制贴合
    }

    result = exporter.export(data, 'test_basic.f3grid', options)
    print(f"\n✓ 导出完成: {result}\n")


def test_with_thin_layer_merge():
    """测试2: 启用薄层合并"""
    print("=" * 70)
    print("测试2: 启用薄层合并")
    print("=" * 70)

    data = create_test_model()
    exporter = F3GridExporterV2()

    options = {
        'downsample_factor': 1,
        'uniform_downsample': True,

        # 启用薄层合并
        'merge_thin_layers': True,
        'merge_thickness_threshold': 0.5,    # 合并 < 0.5m 的层
        'merge_same_lithology_only': True,   # 仅合并同岩性

        'force_layer_continuity': True,
    }

    result = exporter.export(data, 'test_merged.f3grid', options)
    print(f"\n✓ 导出完成: {result}\n")


def test_aggressive_merge():
    """测试3: 激进合并（跨岩性）"""
    print("=" * 70)
    print("测试3: 激进合并（跨岩性）")
    print("=" * 70)

    data = create_test_model()
    exporter = F3GridExporterV2()

    options = {
        'downsample_factor': 1,
        'uniform_downsample': True,

        # 激进合并
        'merge_thin_layers': True,
        'merge_thickness_threshold': 1.0,    # 更大的阈值
        'merge_same_lithology_only': False,  # 跨岩性合并

        'force_layer_continuity': True,
    }

    result = exporter.export(data, 'test_aggressive.f3grid', options)
    print(f"\n✓ 导出完成: {result}\n")


def test_recommended_config():
    """测试4: 推荐配置（最佳稳定性）"""
    print("=" * 70)
    print("测试4: 推荐配置（解决80%问题）")
    print("=" * 70)

    data = create_test_model()
    exporter = F3GridExporterV2()

    # 推荐的稳定性配置
    options = {
        # 基本设置
        'downsample_factor': 1,              # 不降采样
        'uniform_downsample': True,          # 统一降采样
        'min_zone_thickness': 0.001,         # 最小单元厚度

        # 薄层合并（强烈推荐）
        'merge_thin_layers': True,
        'merge_thickness_threshold': 0.5,
        'merge_same_lithology_only': True,

        # 层间贴合（强烈推荐）
        'force_layer_continuity': True,

        # 节点级最小厚度强制（强烈推荐）
        'enforce_minimum_thickness': True,

        # 接口模式
        'create_interfaces': False,          # 层间共享节点
    }

    result = exporter.export(data, 'test_recommended.f3grid', options)
    print(f"\n✓ 导出完成: {result}")
    print("\n" + "=" * 70)
    print("推荐配置测试完成！")
    print("=" * 70)
    print("\n在 FLAC3D 中导入测试:")
    print('  zone import f3grid "test_recommended.f3grid"')
    print('  zone list information')
    print("\n预期结果:")
    print("  ✅ 无 'negative volume' 错误")
    print("  ✅ 无 'tet volumes <= 0' 警告")
    print("  ✅ 所有单元成功导入")
    print("  ✅ 层间完美贴合")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FLAC3D f3grid 导出器 - 完整测试")
    print("=" * 70)
    print("\n测试模型:")
    print("  - 7 个地层（包含 4 个薄煤层）")
    print("  - 网格: 20x20")
    print("  - 薄层厚度: 0.3m - 0.5m")
    print("\n" + "=" * 70 + "\n")

    # 运行所有测试
    test_basic_export()
    test_with_thin_layer_merge()
    test_aggressive_merge()
    test_recommended_config()

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - test_basic.f3grid         (基本导出)")
    print("  - test_merged.f3grid        (薄层合并)")
    print("  - test_aggressive.f3grid    (激进合并)")
    print("  - test_recommended.f3grid   (推荐配置)")
    print("\n请在 FLAC3D 中测试这些文件，比较效果。")
    print("=" * 70)
