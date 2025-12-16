"""
测试重构后的层序建模逻辑
快速验证列重排序和垂向顺序检查是否生效
"""
import sys
import os
import pandas as pd
import numpy as np
from glob import glob

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from layer_modeling import LayerDataProcessor, LayerBasedGeologicalModeling

def main():
    print("=" * 70)
    print("测试重构后的层序建模逻辑")
    print("=" * 70)
    
    # 1. 加载真实数据
    print("\n[1/5] 加载钻孔数据...")
    data_dir = './data'
    csv_files = glob(os.path.join(data_dir, '*.csv'))
    csv_files = [f for f in csv_files if '坐标' not in f]
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            dfs.append(df)
        except:
            df = pd.read_csv(f, encoding='gbk')
            dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  加载了 {len(csv_files)} 个文件，共 {len(df)} 条记录")
    
    # 2. 数据处理
    print("\n[2/5] 处理层序数据...")
    processor = LayerDataProcessor(k_neighbors=10, min_layer_occurrence=1)
    df = processor.standardize_lithology(df)
    processor.infer_layer_order(df)
    thickness_df = processor.extract_thickness_data(df)
    thickness_df = processor.fill_missing_thickness(thickness_df)
    
    print(f"  识别了 {processor.num_layers} 个层")
    print(f"  包括 {len(processor.coal_layers)} 个煤层")
    
    # 3. 构建模型（不使用GNN，只用插值）
    print("\n[3/5] 构建地质模型（使用插值，不使用GNN）...")
    geo_model = LayerBasedGeologicalModeling(
        resolution=(20, 20, 20),  # 小分辨率加快测试
        use_gnn=False,
        smooth_surfaces=True,
        smooth_sigma=1.0,
        min_thickness_floor=0.5  # 强制最小厚度
    )
    
    # 构建模型（这里应该会触发列重排序和垂向检查）
    print("\n开始构建模型...")
    geo_model.build_model(
        df,
        processor,
        trainer=None,
        thickness_df=thickness_df,
        verbose=True  # 确保打印详细信息
    )
    
    # 4. 验证结果
    print("\n[4/5] 验证建模结果...")
    stats = geo_model.get_statistics(processor.layer_order)
    
    # 统计有体积的层
    layers_with_volume = stats[stats['体积 (m³)'] > 0]
    print(f"\n  有体积的层: {len(layers_with_volume)}/{len(stats)}")
    
    if len(layers_with_volume) < 10:
        print("\n  ⚠️ 警告：只有很少的层有体积，可能重构未生效")
        print("\n  前10层统计:")
        print(stats.head(10).to_string(index=False))
    else:
        print("\n  ✅ 成功：大部分层都有体积")
        print("\n  部分层统计（随机10层）:")
        print(stats.sample(min(10, len(stats))).to_string(index=False))
    
    # 5. 导出结果
    print("\n[5/5] 导出结果...")
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    stats.to_csv(os.path.join(output_dir, 'test_refactor_stats.csv'), 
                 index=False, encoding='utf-8-sig')
    print(f"  统计信息已保存: {output_dir}/test_refactor_stats.csv")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
