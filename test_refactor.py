"""
快速验证重构后的层序建模逻辑
"""
import numpy as np

# 模拟场景：3个网格列，5层，存在重叠问题
print("=" * 60)
print("测试列重排序逻辑")
print("=" * 60)

# 构造初始数据（有重叠）
nlay = 5
nx, ny = 3, 3
layer_names = ['16-1煤', '砂岩', '15-4煤', '泥岩', '15-1煤']

# 初始底面和顶面（人工构造有重叠的情况）
bottoms = np.array([
    [[10, 10, 10], [10, 10, 10], [10, 10, 10]],  # 16-1煤
    [[12, 11, 12], [12, 11, 12], [12, 11, 12]],  # 砂岩（与16-1煤重叠！）
    [[20, 20, 20], [20, 20, 20], [20, 20, 20]],  # 15-4煤
    [[22, 22, 22], [22, 22, 22], [22, 22, 22]],  # 泥岩（与15-4煤重叠！）
    [[30, 30, 30], [30, 30, 30], [30, 30, 30]],  # 15-1煤
], dtype=float)

tops = np.array([
    [[13, 13, 13], [13, 13, 13], [13, 13, 13]],  # 16-1煤厚3m
    [[17, 16, 17], [17, 16, 17], [17, 16, 17]],  # 砂岩厚5m
    [[23, 23, 23], [23, 23, 23], [23, 23, 23]],  # 15-4煤厚3m
    [[27, 27, 27], [27, 27, 27], [27, 27, 27]],  # 泥岩厚5m
    [[33, 33, 33], [33, 33, 33], [33, 33, 33]],  # 15-1煤厚3m
], dtype=float)

print("\n初始状态:")
for k in range(nlay):
    print(f"  {layer_names[k]}: bottom[0,0]={bottoms[k,0,0]:.1f}, top[0,0]={tops[k,0,0]:.1f}")

# 检查重叠
print("\n检查重叠:")
overlap_count = 0
for k in range(nlay - 1):
    for i in range(nx):
        for j in range(ny):
            if tops[k, i, j] > bottoms[k+1, i, j]:
                overlap = tops[k, i, j] - bottoms[k+1, i, j]
                print(f"  ❌ [{i},{j}] {layer_names[k]} top={tops[k,i,j]:.1f} > {layer_names[k+1]} bottom={bottoms[k+1,i,j]:.1f} (重叠{overlap:.1f}m)")
                overlap_count += 1

print(f"\n共发现 {overlap_count} 个重叠点")

# 应用列重排序逻辑
print("\n" + "=" * 60)
print("应用列重排序")
print("=" * 60)

min_gap = 0.5
min_thickness = 0.5
fixed_count = 0

for i in range(nx):
    for j in range(ny):
        # 提取这一列
        bcol = bottoms[:, i, j]
        tcol = tops[:, i, j]
        
        # 有效层
        valid_idx = np.where(np.isfinite(bcol) & np.isfinite(tcol))[0]
        if valid_idx.size == 0:
            continue
        
        # 按bottom排序
        order = valid_idx[np.argsort(bcol[valid_idx])]
        
        # 检查是否需要修复
        needs_fix = False
        for ii in range(len(order) - 1):
            if tops[order[ii], i, j] + min_gap > bottoms[order[ii+1], i, j]:
                needs_fix = True
                break
        
        if not needs_fix:
            continue
        
        fixed_count += 1
        
        # 重新码放
        z_cur = float(np.min(bcol[valid_idx]))
        
        for idx in order:
            thick = float(tcol[idx] - bcol[idx])
            if not np.isfinite(thick) or thick < min_thickness:
                thick = min_thickness
            
            bottoms[idx, i, j] = z_cur
            tops[idx, i, j] = z_cur + thick
            z_cur = tops[idx, i, j] + float(min_gap)

print(f"修复了 {fixed_count}/{nx*ny} 个垂直柱")

print("\n修复后状态:")
for k in range(nlay):
    print(f"  {layer_names[k]}: bottom[0,0]={bottoms[k,0,0]:.1f}, top[0,0]={tops[k,0,0]:.1f}")

# 再次检查重叠
print("\n重新检查重叠:")
overlap_count = 0
for k in range(nlay - 1):
    for i in range(nx):
        for j in range(ny):
            if tops[k, i, j] > bottoms[k+1, i, j]:
                overlap = tops[k, i, j] - bottoms[k+1, i, j]
                print(f"  ❌ [{i},{j}] {layer_names[k]} top={tops[k,i,j]:.1f} > {layer_names[k+1]} bottom={bottoms[k+1,i,j]:.1f} (重叠{overlap:.1f}m)")
                overlap_count += 1

if overlap_count == 0:
    print("  ✅ 无重叠，所有层垂向顺序正确！")
else:
    print(f"  ❌ 仍有 {overlap_count} 个重叠点")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
