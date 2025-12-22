"""
诊断 f3grid 文件的节点顺序和几何问题
"""
import numpy as np
import re

def read_f3grid(filepath):
    """读取 f3grid 文件并解析节点和单元"""
    gridpoints = {}  # id -> (x, y, z)
    zones = []  # [(zone_id, [gp_ids])]

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            # 解析节点定义: G <id> <x> <y> <z>
            if line.startswith('G '):
                parts = line.split()
                gp_id = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                gridpoints[gp_id] = (x, y, z)

            # 解析单元定义: Z B8 <id> <gp1> ... <gp8>
            elif line.startswith('Z B8 '):
                parts = line.split()
                zone_id = int(parts[2])
                gp_ids = [int(parts[i]) for i in range(3, 11)]
                zones.append((zone_id, gp_ids))

    return gridpoints, zones


def check_zone_volume(gps, gp_ids):
    """
    检查B8单元的体积（使用四面体分解）

    B8 单元节点顺序应该是：
    底面：0(SW), 1(SE), 2(NW), 3(NE)
    顶面：4(SW), 5(SE), 6(NW), 7(NE)
    """
    # 获取8个节点坐标
    coords = np.array([gps[gid] for gid in gp_ids])

    # B8 单元可以分解为5个四面体
    # 使用中心点分解法
    # 四面体体积公式: V = |det(v1-v0, v2-v0, v3-v0)| / 6

    # 简化检查：计算底面和顶面的中心点
    bottom_center = np.mean(coords[0:4], axis=0)
    top_center = np.mean(coords[4:8], axis=0)

    # 检查厚度（Z方向高度差）
    thickness = top_center[2] - bottom_center[2]

    # 使用一个简单的四面体检查底面4个点是否共面且按正确顺序
    # 底面四点: SW, SE, NW, NE
    p0, p1, p2, p3 = coords[0:4]

    # 向量: SW->SE, SW->NW
    v1 = p1 - p0  # SW to SE (应该指向x正方向)
    v2 = p2 - p0  # SW to NW (应该指向y正方向)

    # 叉积应该指向z正方向
    cross = np.cross(v1, v2)

    # 检查顶面
    p4, p5, p6, p7 = coords[4:8]
    v3 = p5 - p4
    v4 = p6 - p4
    cross_top = np.cross(v3, v4)

    return {
        'thickness': thickness,
        'bottom_cross_z': cross[2],
        'top_cross_z': cross_top[2],
        'bottom_normal': cross / (np.linalg.norm(cross) + 1e-10),
        'top_normal': cross_top / (np.linalg.norm(cross_top) + 1e-10),
    }


def main():
    filepath = r"E:\xiangmu\dizhijianmo\FLAC\geological_model.f3grid"

    print("读取 f3grid 文件...")
    gps, zones = read_f3grid(filepath)

    print(f"总节点数: {len(gps)}")
    print(f"总单元数: {len(zones)}")

    # 检查前10个单元
    print("\n检查前10个单元的几何:")
    for i, (zone_id, gp_ids) in enumerate(zones[:10]):
        result = check_zone_volume(gps, gp_ids)

        print(f"\nZone {zone_id}:")
        print(f"  节点 IDs: {gp_ids}")
        print(f"  厚度: {result['thickness']:.6f} m")
        print(f"  底面法向量Z分量: {result['bottom_cross_z']:.6f}")
        print(f"  顶面法向量Z分量: {result['top_cross_z']:.6f}")
        print(f"  底面归一化法向量: {result['bottom_normal']}")
        print(f"  顶面归一化法向量: {result['top_normal']}")

        # 打印8个节点的坐标
        coords = [gps[gid] for gid in gp_ids]
        print(f"  节点坐标:")
        for j, (gid, coord) in enumerate(zip(gp_ids, coords)):
            label = ['SW_bot', 'SE_bot', 'NW_bot', 'NE_bot',
                     'SW_top', 'SE_top', 'NW_top', 'NE_top'][j]
            print(f"    [{j}] {label} (GP#{gid}): ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")

        # 检查问题
        issues = []
        if result['thickness'] <= 0:
            issues.append("❌ 负厚度或零厚度")
        if result['bottom_cross_z'] < 0:
            issues.append("❌ 底面法向量指向下方（节点顺序错误）")
        if result['top_cross_z'] < 0:
            issues.append("❌ 顶面法向量指向下方（节点顺序错误）")

        if issues:
            print(f"  ⚠️  发现问题:")
            for issue in issues:
                print(f"      {issue}")
        else:
            print(f"  ✓ 几何正常")

    # 统计有问题的单元
    print("\n" + "="*60)
    print("统计分析:")

    negative_thickness = 0
    negative_bottom_cross = 0
    negative_top_cross = 0

    for zone_id, gp_ids in zones:
        result = check_zone_volume(gps, gp_ids)
        if result['thickness'] <= 0:
            negative_thickness += 1
        if result['bottom_cross_z'] < 0:
            negative_bottom_cross += 1
        if result['top_cross_z'] < 0:
            negative_top_cross += 1

    print(f"负厚度单元数: {negative_thickness} / {len(zones)} ({100*negative_thickness/len(zones):.1f}%)")
    print(f"底面法向量错误: {negative_bottom_cross} / {len(zones)} ({100*negative_bottom_cross/len(zones):.1f}%)")
    print(f"顶面法向量错误: {negative_top_cross} / {len(zones)} ({100*negative_top_cross/len(zones):.1f}%)")


if __name__ == '__main__':
    main()
