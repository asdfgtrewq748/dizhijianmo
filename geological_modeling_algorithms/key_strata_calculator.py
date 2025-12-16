# backend/key_strata_calculator.py
"""
关键层计算模块 - 重构版本
将巨型函数拆分为多个小函数,提高可维护性和性能
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import warnings


class KeyStrataCalculator:
    """关键层计算器 - 面向对象重构版本"""

    def __init__(self):
        """初始化计算器"""
        self.required_cols = ['厚度/m', '弹性模量/GPa', '容重/kN·m-3', '抗拉强度/MPa']

    def calculate(self, df_strata_above_coal: pd.DataFrame,
                  coal_seam_properties_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        计算给定煤层上覆岩层的关键层信息

        Args:
            df_strata_above_coal: 煤层上方的岩层数据
            coal_seam_properties_df: 煤层属性数据

        Returns:
            关键层信息列表
        """
        if df_strata_above_coal.empty or coal_seam_properties_df.empty:
            return []

        try:
            # 1. 数据准备和验证
            df_prepared = self._prepare_data(df_strata_above_coal)

            # 2. 提取煤层采高
            mining_height = self._get_mining_height(coal_seam_properties_df)
            mining_height_factor = mining_height * 1.4

            # 3. 计算关键物理量
            eh_values, rh_values = self._calculate_eh_rh(df_prepared)

            # 4. 识别关键层
            key_flags = self._identify_key_strata(eh_values, rh_values)

            # 5. 应用特殊规则(如直接顶检查、泥岩过滤)
            key_flags = self._apply_special_rules(
                key_flags, df_prepared, mining_height_factor
            )

            # 6. 标记主关键层(PKS)
            key_flags = self._mark_primary_key_stratum(
                key_flags, eh_values, rh_values, df_prepared
            )

            # 7. 生成输出结果
            result = self._format_output(key_flags, df_prepared)

            return result

        except Exception as e:
            warnings.warn(f"计算关键层时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备和验证输入数据

        Args:
            df: 原始岩层数据

        Returns:
            准备好的数据框
        """
        df = df.copy()

        # 确保所有必需列存在
        for col in self.required_cols:
            if col not in df.columns:
                df[col] = 0.0

        # 转换为数值类型
        for col in self.required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 填充缺失值
        df.fillna(0, inplace=True)

        # 确保岩层名称列存在
        if '岩层名称' not in df.columns:
            df['岩层名称'] = f'岩层{df.index + 1}'

        return df

    def _get_mining_height(self, coal_df: pd.DataFrame) -> float:
        """
        获取煤层采高

        Args:
            coal_df: 煤层属性数据框

        Returns:
            采高值(米)
        """
        mining_height_val = pd.to_numeric(
            coal_df['厚度/m'].iloc[0], errors='coerce'
        )
        mining_height = round(float(mining_height_val), 2) if pd.notna(mining_height_val) else 0.0
        return mining_height

    def _calculate_eh_rh(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算EH和RH值(弹性特性和容重特性)

        Args:
            df: 岩层数据框

        Returns:
            (eh_flipped, rh_flipped) - 翻转后的数组(从煤层向上)
        """
        thickness = df['厚度/m'].values
        elastic_modulus = df['弹性模量/GPa'].values
        bulk_density = df['容重/kN·m-3'].values

        # RH = 容重 × 厚度
        rh_orig = bulk_density * thickness

        # EH = 弹性模量 × 厚度³
        eh_orig = elastic_modulus * (thickness ** 3)

        # 翻转数组(从煤层向上排列)
        eh_flipped = np.flipud(eh_orig)
        rh_flipped = np.flipud(rh_orig)

        return eh_flipped, rh_flipped

    def _identify_key_strata(self, eh_values: np.ndarray,
                            rh_values: np.ndarray,
                            max_iterations: int = 500) -> np.ndarray:
        """
        识别关键层

        算法原理:
        - 计算每层的荷载传递系数 q(x)
        - 当 q(x) 出现下降时,表明找到了关键层
        - 迭代处理,直到识别所有关键层

        Args:
            eh_values: EH值数组
            rh_values: RH值数组
            max_iterations: 最大迭代次数

        Returns:
            关键层标记数组(1表示关键层,0表示非关键层)
        """
        n = len(rh_values)
        key_flags = np.zeros(n, dtype=int)

        # 工作副本
        temp_rh = rh_values.copy()
        temp_eh = eh_values.copy()

        # 记录已删除的行数
        deleted_rows = 0

        for iteration in range(max_iterations):
            if len(temp_rh) == 0 or len(temp_eh) == 0:
                break

            # 计算荷载传递系数 q(x)
            q_x = self._calculate_load_transfer_coefficient(temp_rh, temp_eh)

            # 寻找关键层(q(x)首次下降的位置)
            key_found, key_idx = self._find_key_layer_position(q_x)

            if key_found:
                # 标记关键层
                original_idx = key_idx - 1 + deleted_rows
                if 0 <= original_idx < n:
                    key_flags[original_idx] = 1

                # 移除已处理的层
                temp_rh = temp_rh[key_idx:]
                temp_eh = temp_eh[key_idx:]
                deleted_rows += key_idx
            else:
                # 没有找到更多关键层
                # 如果还没有任何关键层,将第一层标记为关键层
                if not np.any(key_flags) and len(temp_rh) > 0:
                    key_flags[deleted_rows] = 1
                break

        return key_flags

    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_load_transfer_coefficient_cached(rh_tuple: tuple, eh_tuple: tuple) -> tuple:
        """
        缓存版本的荷载传递系数计算(用于相同输入)

        Args:
            rh_tuple, eh_tuple: 元组形式的输入(用于哈希缓存)

        Returns:
            q(x)数组的元组形式
        """
        rh = np.array(rh_tuple)
        eh = np.array(eh_tuple)

        q_x = np.zeros(len(rh))
        for i in range(len(rh)):
            sum_rh_slice = np.sum(rh[:i+1])
            sum_eh_slice = np.sum(eh[:i+1])
            if sum_eh_slice != 0:
                q_x[i] = eh[0] * sum_rh_slice / sum_eh_slice
            else:
                q_x[i] = 0

        return tuple(q_x)

    def _calculate_load_transfer_coefficient(self, rh: np.ndarray,
                                            eh: np.ndarray) -> np.ndarray:
        """
        计算荷载传递系数 q(x)

        q(x) = EH₁ × Σ(RH) / Σ(EH)

        Args:
            rh: RH值数组
            eh: EH值数组

        Returns:
            q(x)数组
        """
        # 使用NumPy向量化操作优化性能
        n = len(rh)
        q_x = np.zeros(n)

        # 预计算累积和(性能优化)
        cumsum_rh = np.cumsum(rh)
        cumsum_eh = np.cumsum(eh)

        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            q_x = np.where(
                cumsum_eh != 0,
                eh[0] * cumsum_rh / cumsum_eh,
                0
            )

        return q_x

    @staticmethod
    def _find_key_layer_position(q_x: np.ndarray) -> Tuple[bool, int]:
        """
        找到关键层位置(q(x)首次下降的位置)

        Args:
            q_x: 荷载传递系数数组

        Returns:
            (是否找到, 位置索引)
        """
        for i in range(1, len(q_x)):
            if q_x[i] < q_x[i-1]:
                return True, i
        return False, -1

    def _apply_special_rules(self, key_flags: np.ndarray,
                            df: pd.DataFrame,
                            mining_height_factor: float) -> np.ndarray:
        """
        应用特殊规则

        规则1: 直接顶检查 - 如果直接顶厚度 > 采高×1.4,且到第一关键层距离>10m,
              则将直接顶也标记为关键层
        规则2: 泥岩过滤 - 泥岩不应作为关键层

        Args:
            key_flags: 关键层标记数组
            df: 岩层数据框
            mining_height_factor: 采高因子

        Returns:
            更新后的关键层标记数组
        """
        key_flags = key_flags.copy()

        # 翻转数据以匹配从煤层向上的顺序
        thickness_flipped = np.flipud(df['厚度/m'].values)
        names_flipped = np.flipud(df['岩层名称'].values)

        # 规则1: 直接顶检查
        first_key_indices = np.where(key_flags == 1)[0]
        if len(first_key_indices) > 0:
            first_key_idx = first_key_indices[0]
            if first_key_idx > 0:
                immediate_roof_thickness = thickness_flipped[0]
                if immediate_roof_thickness > mining_height_factor:
                    # 计算到第一关键层的累计厚度
                    cumsum_thickness = np.sum(thickness_flipped[:first_key_idx+1])
                    if cumsum_thickness > 10:
                        key_flags[0] = 1

        # 规则2: 泥岩过滤
        for i, name in enumerate(names_flipped):
            if '泥岩' in str(name):
                key_flags[i] = 0

        return key_flags

    def _mark_primary_key_stratum(self, key_flags: np.ndarray,
                                  eh_values: np.ndarray,
                                  rh_values: np.ndarray,
                                  df: pd.DataFrame) -> np.ndarray:
        """
        标记主关键层(PKS - Primary Key Stratum)

        主关键层是极限跨距L(x)最大的关键层
        L(x) = h × √(2Rt / q(z))

        Args:
            key_flags: 关键层标记数组
            eh_values: EH值数组
            rh_values: RH值数组
            df: 岩层数据框

        Returns:
            更新后的关键层标记数组(主关键层标记为2)
        """
        key_indices = np.where(key_flags == 1)[0]
        if len(key_indices) == 0:
            return key_flags

        # 获取关键层的物理参数
        thickness_flipped = np.flipud(df['厚度/m'].values)
        tensile_strength_flipped = np.flipud(df['抗拉强度/MPa'].values)

        # 计算每个关键层的 q(z) 值
        q_z_values = self._calculate_qz_for_key_strata(
            key_indices, eh_values, rh_values
        )

        # 获取关键层参数
        h_values = thickness_flipped[key_indices]
        rt_values = tensile_strength_flipped[key_indices]

        # 转换q(z)为MPa单位
        q_z_mpa = q_z_values / 1000.0

        # 计算极限跨距 L(x) = h × √(2Rt / q(z))
        l_x = self._calculate_limit_span(h_values, rt_values, q_z_mpa)

        # 找到L(x)最大的关键层作为主关键层
        if len(l_x) > 0 and np.any(np.isfinite(l_x)) and np.any(l_x > 0):
            pks_idx_in_lx_array = np.nanargmax(l_x)
            pks_original_idx = key_indices[pks_idx_in_lx_array]
            key_flags[pks_original_idx] = 2  # 2表示主关键层

        return key_flags

    def _calculate_qz_for_key_strata(self, key_indices: np.ndarray,
                                     eh_values: np.ndarray,
                                     rh_values: np.ndarray) -> np.ndarray:
        """
        计算关键层的q(z)值

        q(z) = EH_key × Σ(RH_above) / Σ(EH_above)

        Args:
            key_indices: 关键层索引数组
            eh_values: EH值数组
            rh_values: RH值数组

        Returns:
            q(z)值数组
        """
        q_z_values = np.zeros(len(key_indices))

        for i, key_idx in enumerate(key_indices):
            if i == 0:
                # 第一个关键层:从煤层到该层
                sum_rh = np.sum(rh_values[:key_idx + 1])
                sum_eh = np.sum(eh_values[:key_idx + 1])
            else:
                # 后续关键层:从上一个关键层到当前层
                prev_key_idx = key_indices[i-1]
                sum_rh = np.sum(rh_values[prev_key_idx + 1 : key_idx + 1])
                sum_eh = np.sum(eh_values[prev_key_idx + 1 : key_idx + 1])

            if sum_eh != 0:
                q_z_values[i] = eh_values[key_idx] * sum_rh / sum_eh
            else:
                q_z_values[i] = 0

        return q_z_values

    @staticmethod
    def _calculate_limit_span(h: np.ndarray, rt: np.ndarray,
                              q_z_mpa: np.ndarray) -> np.ndarray:
        """
        计算极限跨距 L(x)

        L(x) = h × √(2Rt / q(z))

        Args:
            h: 厚度数组
            rt: 抗拉强度数组
            q_z_mpa: q(z)值数组(MPa单位)

        Returns:
            极限跨距数组
        """
        l_x = np.zeros_like(h, dtype=float)

        # 只计算q_z > 0的情况
        valid_mask = q_z_mpa > 0

        if np.any(valid_mask):
            h_subset = h[valid_mask]
            rt_subset = rt[valid_mask]
            q_z_subset = q_z_mpa[valid_mask]

            # 计算 √(2Rt / q(z))
            term_in_sqrt = (2 * rt_subset) / q_z_subset
            # 确保开方项非负
            safe_term = np.where(term_in_sqrt >= 0, term_in_sqrt, 0)

            l_x[valid_mask] = h_subset * np.sqrt(safe_term)

        return l_x

    def _format_output(self, key_flags: np.ndarray,
                       df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        格式化输出结果

        Args:
            key_flags: 关键层标记数组(0=非关键层, 1=关键层, 2=主关键层)
            df: 岩层数据框

        Returns:
            关键层信息列表
        """
        result = []

        # 翻转数据以匹配从煤层向上的顺序
        thickness_flipped = np.flipud(df['厚度/m'].values)
        names_flipped = np.flipud(df['岩层名称'].values)

        # 生成SK标签
        sk_labels = self._generate_sk_labels(key_flags)

        # 计算从煤层的累计距离
        cumulative_thickness = 0.0

        for i in range(len(key_flags)):
            if key_flags[i] > 0:  # 关键层或主关键层
                current_thickness = float(thickness_flipped[i])
                distance_from_coal = round(
                    cumulative_thickness + current_thickness / 2, 2
                )

                entry = {
                    '岩性': names_flipped[i],
                    '厚度': round(current_thickness, 2),
                    '距煤层距离': distance_from_coal,
                    'SK_Label': sk_labels[i]
                }
                result.append(entry)

            cumulative_thickness += float(thickness_flipped[i])

        return result

    @staticmethod
    def _generate_sk_labels(key_flags: np.ndarray) -> List[str]:
        """
        生成SK标签(SK1, SK2, SK3(PKS)等)

        Args:
            key_flags: 关键层标记数组(0=非关键层, 1=关键层, 2=主关键层)

        Returns:
            标签列表
        """
        labels = ['-'] * len(key_flags)
        sk_count = 1

        for i, flag in enumerate(key_flags):
            if flag == 1:
                labels[i] = f'SK{sk_count}'
                sk_count += 1
            elif flag == 2:
                labels[i] = f'SK{sk_count}(PKS)'
                sk_count += 1

        return labels


# 便捷函数 - 保持向后兼容
def calculate_key_strata_details(df_strata_above_coal: pd.DataFrame,
                                 coal_seam_properties_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    计算关键层详情(向后兼容的便捷函数)

    Args:
        df_strata_above_coal: 煤层上方的岩层数据
        coal_seam_properties_df: 煤层属性数据

    Returns:
        关键层信息列表
    """
    calculator = KeyStrataCalculator()
    return calculator.calculate(df_strata_above_coal, coal_seam_properties_df)
