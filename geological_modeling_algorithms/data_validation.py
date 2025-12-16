# backend/data_validation.py
"""
数据验证模块 - 业务规则验证和数据一致性检查
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ValidationRule:
    """验证规则"""
    field: str
    rule_type: str  # 'range', 'pattern', 'custom'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pattern: Optional[str] = None
    custom_func: Optional[callable] = None
    error_message: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_count: int = 0

    def add_error(self, message: str):
        """添加错误"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """添加警告"""
        self.warnings.append(message)


class GeologicalDataValidator:
    """地质数据验证器"""

    # 业务规则定义
    RULES = {
        "厚度/m": ValidationRule(
            field="厚度/m",
            rule_type="range",
            min_val=0.0,
            max_val=1000.0,
            error_message="厚度必须在0-1000米之间"
        ),
        "弹性模量/GPa": ValidationRule(
            field="弹性模量/GPa",
            rule_type="range",
            min_val=0.1,
            max_val=100.0,
            error_message="弹性模量必须在0.1-100 GPa之间"
        ),
        "容重/kN·m-3": ValidationRule(
            field="容重/kN·m-3",
            rule_type="range",
            min_val=10.0,
            max_val=30.0,
            error_message="容重必须在10-30 kN/m³之间"
        ),
        "抗拉强度/MPa": ValidationRule(
            field="抗拉强度/MPa",
            rule_type="range",
            min_val=0.1,
            max_val=50.0,
            error_message="抗拉强度必须在0.1-50 MPa之间"
        ),
    }

    # 中国境内坐标范围 (大致范围)
    CHINA_BOUNDS = {
        "longitude": (73.0, 135.0),  # 经度
        "latitude": (18.0, 54.0),     # 纬度
        "x_planar": (200000, 800000),  # 平面坐标X (米,具体取决于投影)
        "y_planar": (2000000, 6000000)  # 平面坐标Y (米)
    }

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: 严格模式,在严格模式下错误会抛出异常
        """
        self.strict_mode = strict_mode

    def validate_dataframe(self, df: pd.DataFrame,
                          auto_fix: bool = True) -> ValidationResult:
        """
        验证DataFrame

        Args:
            df: 要验证的数据框
            auto_fix: 是否自动修复可修复的问题

        Returns:
            验证结果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        if df.empty:
            result.add_error("数据框为空")
            return result

        # 1. 检查必需列
        required_cols = ["岩层名称", "厚度/m"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result.add_error(f"缺少必需列: {', '.join(missing_cols)}")
            return result

        # 2. 验证数值范围
        for col_name, rule in self.RULES.items():
            if col_name in df.columns:
                col_result = self._validate_column_range(
                    df[col_name], rule, auto_fix
                )
                result.errors.extend(col_result.errors)
                result.warnings.extend(col_result.warnings)
                result.fixed_count += col_result.fixed_count
                if not col_result.is_valid:
                    result.is_valid = False

        # 3. 检查数据一致性
        consistency_result = self._check_data_consistency(df)
        result.errors.extend(consistency_result.errors)
        result.warnings.extend(consistency_result.warnings)

        # 4. 检测重复数据
        duplicate_result = self._check_duplicates(df)
        result.warnings.extend(duplicate_result.warnings)

        # 5. 检查缺失值
        missing_result = self._check_missing_values(df)
        result.warnings.extend(missing_result.warnings)

        return result

    def _validate_column_range(self, series: pd.Series,
                               rule: ValidationRule,
                               auto_fix: bool) -> ValidationResult:
        """验证列的数值范围"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # 转换为数值类型
        numeric_series = pd.to_numeric(series, errors='coerce')

        # 检查非数值
        non_numeric_count = numeric_series.isna().sum() - series.isna().sum()
        if non_numeric_count > 0:
            result.add_warning(
                f"列 '{rule.field}' 包含 {non_numeric_count} 个非数值项"
            )

        # 过滤掉NaN
        valid_values = numeric_series.dropna()
        if len(valid_values) == 0:
            return result

        # 检查范围
        if rule.min_val is not None:
            below_min = valid_values < rule.min_val
            if below_min.any():
                count = below_min.sum()
                min_found = valid_values[below_min].min()
                if auto_fix:
                    result.add_warning(
                        f"列 '{rule.field}': {count} 个值小于最小值 {rule.min_val} "
                        f"(最小发现值: {min_found:.2f}), 已自动修正"
                    )
                    result.fixed_count += count
                else:
                    result.add_error(
                        f"列 '{rule.field}': {count} 个值小于最小值 {rule.min_val} "
                        f"(最小发现值: {min_found:.2f})"
                    )

        if rule.max_val is not None:
            above_max = valid_values > rule.max_val
            if above_max.any():
                count = above_max.sum()
                max_found = valid_values[above_max].max()
                if auto_fix:
                    result.add_warning(
                        f"列 '{rule.field}': {count} 个值超过最大值 {rule.max_val} "
                        f"(最大发现值: {max_found:.2f}), 已自动修正"
                    )
                    result.fixed_count += count
                else:
                    result.add_error(
                        f"列 '{rule.field}': {count} 个值超过最大值 {rule.max_val} "
                        f"(最大发现值: {max_found:.2f})"
                    )

        return result

    def _check_data_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """检查数据一致性"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # 检查1: 泥岩的强度应该较低
        if all(col in df.columns for col in ["岩层名称", "抗拉强度/MPa"]):
            mudstone_mask = df["岩层名称"].astype(str).str.contains("泥岩", na=False)
            if mudstone_mask.any():
                mudstone_strength = pd.to_numeric(
                    df.loc[mudstone_mask, "抗拉强度/MPa"],
                    errors='coerce'
                )
                high_strength = mudstone_strength > 5.0  # 泥岩抗拉强度通常<5MPa
                if high_strength.any():
                    result.add_warning(
                        f"发现 {high_strength.sum()} 个泥岩层的抗拉强度异常高 (>5 MPa)"
                    )

        # 检查2: 砂岩的密度应该合理
        if all(col in df.columns for col in ["岩层名称", "容重/kN·m-3"]):
            sandstone_mask = df["岩层名称"].astype(str).str.contains("砂岩", na=False)
            if sandstone_mask.any():
                sandstone_density = pd.to_numeric(
                    df.loc[sandstone_mask, "容重/kN·m-3"],
                    errors='coerce'
                )
                # 砂岩容重通常在22-27 kN/m³
                abnormal = (sandstone_density < 20) | (sandstone_density > 28)
                if abnormal.any():
                    result.add_warning(
                        f"发现 {abnormal.sum()} 个砂岩层的容重异常 (正常范围: 20-28 kN/m³)"
                    )

        # 检查3: 煤层的厚度应该合理
        if all(col in df.columns for col in ["岩层名称", "厚度/m"]):
            coal_mask = df["岩层名称"].astype(str).str.contains("煤", na=False)
            if coal_mask.any():
                coal_thickness = pd.to_numeric(
                    df.loc[coal_mask, "厚度/m"],
                    errors='coerce'
                )
                # 煤层厚度通常在0.5-20米
                very_thin = coal_thickness < 0.3
                very_thick = coal_thickness > 20
                if very_thin.any():
                    result.add_warning(
                        f"发现 {very_thin.sum()} 个煤层厚度过薄 (<0.3m), 可能不具开采价值"
                    )
                if very_thick.any():
                    result.add_warning(
                        f"发现 {very_thick.sum()} 个煤层厚度过大 (>20m), 请确认数据准确性"
                    )

        return result

    def _check_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """检查重复数据"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # 检查完全重复的行
        duplicate_rows = df.duplicated()
        if duplicate_rows.any():
            count = duplicate_rows.sum()
            result.add_warning(f"发现 {count} 行完全重复的数据")

        # 如果有钻孔名和岩层名称,检查同一钻孔中的重复岩层
        if all(col in df.columns for col in ["钻孔名", "岩层名称"]):
            duplicate_layers = df.duplicated(subset=["钻孔名", "岩层名称"])
            if duplicate_layers.any():
                count = duplicate_layers.sum()
                result.add_warning(
                    f"发现 {count} 个钻孔中存在重复的岩层记录"
                )

        return result

    def _check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """检查缺失值"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100
                if missing_pct > 50:
                    result.add_warning(
                        f"列 '{col}' 缺失值过多: {missing_count} ({missing_pct:.1f}%)"
                    )
                elif missing_pct > 10:
                    result.add_warning(
                        f"列 '{col}' 包含 {missing_count} 个缺失值 ({missing_pct:.1f}%)"
                    )

        return result

    def validate_coordinates(self, x: np.ndarray, y: np.ndarray,
                            coord_type: str = 'planar') -> ValidationResult:
        """
        验证坐标是否在合理范围内

        Args:
            x, y: 坐标数组
            coord_type: 坐标类型 ('planar' 平面坐标, 'geographic' 地理坐标)

        Returns:
            验证结果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        if coord_type == 'geographic':
            # 地理坐标 (经纬度)
            lon_min, lon_max = self.CHINA_BOUNDS['longitude']
            lat_min, lat_max = self.CHINA_BOUNDS['latitude']

            out_of_bounds_x = (x < lon_min) | (x > lon_max)
            out_of_bounds_y = (y < lat_min) | (y > lat_max)

            if out_of_bounds_x.any():
                count = out_of_bounds_x.sum()
                result.add_warning(
                    f"{count} 个点的经度超出中国范围 ({lon_min}° - {lon_max}°)"
                )

            if out_of_bounds_y.any():
                count = out_of_bounds_y.sum()
                result.add_warning(
                    f"{count} 个点的纬度超出中国范围 ({lat_min}° - {lat_max}°)"
                )

        elif coord_type == 'planar':
            # 平面坐标 (投影坐标)
            x_min, x_max = self.CHINA_BOUNDS['x_planar']
            y_min, y_max = self.CHINA_BOUNDS['y_planar']

            # 检查坐标范围
            if x.min() < 0 or y.min() < 0:
                result.add_warning("发现负坐标值,请确认坐标系统")

            # 检查坐标是否过于分散
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()

            if x_range > 1000000 or y_range > 1000000:
                result.add_warning(
                    f"坐标范围过大 (X: {x_range:.0f}m, Y: {y_range:.0f}m), "
                    "请确认坐标单位和投影系统"
                )

        return result

    def check_borehole_depth_consistency(self, df: pd.DataFrame,
                                         borehole_col: str = "钻孔名",
                                         thickness_col: str = "厚度/m") -> ValidationResult:
        """
        检查同一钻孔的岩层累计厚度一致性

        Args:
            df: 数据框
            borehole_col: 钻孔名称列
            thickness_col: 厚度列

        Returns:
            验证结果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        if borehole_col not in df.columns or thickness_col not in df.columns:
            return result

        # 按钻孔分组
        grouped = df.groupby(borehole_col)[thickness_col].apply(
            lambda x: pd.to_numeric(x, errors='coerce').sum()
        )

        # 检查异常深度
        median_depth = grouped.median()
        std_depth = grouped.std()

        if std_depth > median_depth * 0.5:
            result.add_warning(
                f"钻孔深度差异较大 (中位数: {median_depth:.1f}m, "
                f"标准差: {std_depth:.1f}m)"
            )

        # 列出异常钻孔
        if median_depth > 0 and std_depth > 0:
            z_scores = np.abs((grouped - median_depth) / std_depth)
            abnormal_boreholes = grouped[z_scores > 3]

            if len(abnormal_boreholes) > 0:
                result.add_warning(
                    f"发现 {len(abnormal_boreholes)} 个深度异常的钻孔: "
                    f"{', '.join(abnormal_boreholes.index[:5].tolist())}"
                    + ("..." if len(abnormal_boreholes) > 5 else "")
                )

        return result


def validate_geological_data(df: pd.DataFrame,
                             strict: bool = False,
                             auto_fix: bool = True) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    验证地质数据的便捷函数

    Args:
        df: 要验证的数据框
        strict: 严格模式
        auto_fix: 自动修复

    Returns:
        (处理后的数据框, 验证结果)
    """
    validator = GeologicalDataValidator(strict_mode=strict)
    result = validator.validate_dataframe(df, auto_fix=auto_fix)

    # 如果启用自动修复,应用修复
    if auto_fix and result.fixed_count > 0:
        df = df.copy()
        for col_name, rule in validator.RULES.items():
            if col_name in df.columns:
                numeric_col = pd.to_numeric(df[col_name], errors='coerce')
                if rule.min_val is not None:
                    numeric_col = numeric_col.clip(lower=rule.min_val)
                if rule.max_val is not None:
                    numeric_col = numeric_col.clip(upper=rule.max_val)
                df[col_name] = numeric_col

    return df, result
