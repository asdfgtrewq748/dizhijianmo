"""
统计分析模块
提供科研级统计分析功能，包括描述性统计、相关性分析、回归分析等
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class DescriptiveStatistics:
    """描述性统计分析"""
    
    @staticmethod
    def calculate_basic_stats(data: pd.Series) -> Dict[str, float]:
        """
        计算基础统计指标
        
        Args:
            data: 数据序列
            
        Returns:
            包含各项统计指标的字典
        """
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            return {
                'count': 0,
                'missing': len(data),
                'error': '无有效数据'
            }
        
        try:
            result = {
                # 基本信息
                'count': int(len(data_clean)),
                'missing': int(data.isna().sum()),
                'missing_rate': float(data.isna().sum() / len(data) * 100),
                
                # 集中趋势
                'mean': float(data_clean.mean()),
                'median': float(data_clean.median()),
                'mode': float(data_clean.mode().iloc[0]) if len(data_clean.mode()) > 0 else None,
                
                # 离散程度
                'std': float(data_clean.std()),
                'variance': float(data_clean.var()),
                'range': float(data_clean.max() - data_clean.min()),
                'iqr': float(data_clean.quantile(0.75) - data_clean.quantile(0.25)),
                'cv': float(data_clean.std() / data_clean.mean() * 100) if data_clean.mean() != 0 else None,
                
                # 极值
                'min': float(data_clean.min()),
                'max': float(data_clean.max()),
                
                # 分位数
                'q25': float(data_clean.quantile(0.25)),
                'q50': float(data_clean.quantile(0.50)),
                'q75': float(data_clean.quantile(0.75)),
                
                # 形态特征
                'skewness': float(data_clean.skew()),
                'kurtosis': float(data_clean.kurtosis()),
                
                # 标准误差
                'sem': float(data_clean.sem()),
                
                # 置信区间（95%）
                'ci_lower': float(data_clean.mean() - 1.96 * data_clean.sem()),
                'ci_upper': float(data_clean.mean() + 1.96 * data_clean.sem()),
            }
            
            return result
        except Exception as e:
            return {
                'count': int(len(data_clean)),
                'missing': int(data.isna().sum()),
                'error': str(e)
            }
    
    @staticmethod
    def calculate_summary_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算多列数据的统计摘要表
        
        Args:
            df: 数据框
            columns: 要分析的列名列表，None表示所有数值列
            
        Returns:
            统计摘要表
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary_data = []
        for col in columns:
            stats = DescriptiveStatistics.calculate_basic_stats(df[col])
            stats['variable'] = col
            summary_data.append(stats)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 重新排序列
        cols_order = ['variable', 'count', 'missing', 'mean', 'std', 'min', 'q25', 
                      'median', 'q75', 'max', 'skewness', 'kurtosis']
        existing_cols = [col for col in cols_order if col in summary_df.columns]
        
        return summary_df[existing_cols]


class CorrelationAnalysis:
    """相关性分析"""
    
    @staticmethod
    def calculate_correlation_matrix(
        df: pd.DataFrame, 
        method: str = 'pearson',
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        计算相关系数矩阵
        
        Args:
            df: 数据框
            method: 'pearson' 或 'spearman'
            columns: 要分析的列名列表
            
        Returns:
            (相关系数矩阵, p值矩阵)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        data = df[columns].dropna()
        n = len(columns)
        
        corr_matrix = np.zeros((n, n))
        pvalue_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pvalue_matrix[i, j] = 0.0
                else:
                    try:
                        if method == 'pearson':
                            corr, pval = pearsonr(data[columns[i]], data[columns[j]])
                        else:
                            corr, pval = spearmanr(data[columns[i]], data[columns[j]])
                        
                        corr_matrix[i, j] = corr
                        pvalue_matrix[i, j] = pval
                    except Exception:
                        corr_matrix[i, j] = np.nan
                        pvalue_matrix[i, j] = np.nan
        
        corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        pvalue_df = pd.DataFrame(pvalue_matrix, index=columns, columns=columns)
        
        return corr_df, pvalue_df
    
    @staticmethod
    def get_significant_correlations(
        corr_df: pd.DataFrame, 
        pvalue_df: pd.DataFrame, 
        threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        获取显著相关的变量对
        
        Args:
            corr_df: 相关系数矩阵
            pvalue_df: p值矩阵
            threshold: 显著性水平
            
        Returns:
            显著相关对列表
        """
        significant_pairs = []
        
        for i in range(len(corr_df)):
            for j in range(i + 1, len(corr_df)):
                corr = corr_df.iloc[i, j]
                pval = pvalue_df.iloc[i, j]
                
                if not np.isnan(corr) and pval < threshold:
                    significant_pairs.append({
                        'var1': corr_df.index[i],
                        'var2': corr_df.columns[j],
                        'correlation': float(corr),
                        'p_value': float(pval),
                        'strength': CorrelationAnalysis._classify_strength(abs(corr))
                    })
        
        # 按相关系数绝对值排序
        significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_pairs
    
    @staticmethod
    def _classify_strength(corr: float) -> str:
        """相关性强度分类"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return '极强'
        elif abs_corr >= 0.6:
            return '强'
        elif abs_corr >= 0.4:
            return '中等'
        elif abs_corr >= 0.2:
            return '弱'
        else:
            return '极弱'


class RegressionAnalysis:
    """回归分析"""
    
    @staticmethod
    def linear_regression(
        x: np.ndarray, 
        y: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        线性回归分析
        
        Args:
            x: 自变量
            y: 因变量
            confidence_level: 置信水平
            
        Returns:
            回归结果字典
        """
        # 数据清洗
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return {'error': '有效数据点不足'}
        
        # 拟合模型
        model = LinearRegression()
        model.fit(x_clean, y_clean)
        
        # 预测
        y_pred = model.predict(x_clean)
        
        # 计算统计量
        n = len(x_clean)
        k = 1  # 自变量个数
        
        # R² 和调整后的 R²
        r2 = r2_score(y_clean, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        # MSE 和 RMSE
        mse = mean_squared_error(y_clean, y_pred)
        rmse = np.sqrt(mse)
        
        # 残差分析
        residuals = y_clean - y_pred
        
        # F统计量和p值
        ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        ss_regression = ss_total - ss_residual
        
        f_statistic = (ss_regression / k) / (ss_residual / (n - k - 1)) if ss_residual > 0 else np.inf
        p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
        
        # 置信区间
        alpha = 1 - confidence_level
        t_val = stats.t.ppf(1 - alpha / 2, n - 2)
        
        # 斜率标准误
        se = np.sqrt(mse / np.sum((x_clean.flatten() - np.mean(x_clean)) ** 2))
        slope_ci = [model.coef_[0] - t_val * se, model.coef_[0] + t_val * se]
        
        # 预测区间
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
        y_range_pred = model.predict(x_range)
        
        # 计算预测区间
        x_mean = np.mean(x_clean)
        sxx = np.sum((x_clean - x_mean) ** 2)
        se_pred = np.sqrt(mse * (1 + 1/n + (x_range.flatten() - x_mean)**2 / sxx))
        
        pred_interval_lower = y_range_pred - t_val * se_pred
        pred_interval_upper = y_range_pred + t_val * se_pred
        
        return {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r_squared': float(r2),
            'adj_r_squared': float(adj_r2),
            'rmse': float(rmse),
            'mse': float(mse),
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'n_observations': int(n),
            'slope_ci': [float(slope_ci[0]), float(slope_ci[1])],
            'equation': f'y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}',
            'residuals': residuals.tolist(),
            'fitted_values': y_pred.tolist(),
            'prediction_range': {
                'x': x_range.flatten().tolist(),
                'y': y_range_pred.tolist(),
                'lower': pred_interval_lower.tolist(),
                'upper': pred_interval_upper.tolist()
            }
        }
    
    @staticmethod
    def polynomial_regression(
        x: np.ndarray, 
        y: np.ndarray,
        degree: int = 2
    ) -> Dict[str, Any]:
        """
        多项式回归分析
        
        Args:
            x: 自变量
            y: 因变量
            degree: 多项式阶数
            
        Returns:
            回归结果字典
        """
        # 数据清洗
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask].reshape(-1, 1)
        y_clean = y[mask]
        
        if len(x_clean) < degree + 2:
            return {'error': '有效数据点不足'}
        
        # 生成多项式特征
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x_clean)
        
        # 拟合模型
        model = LinearRegression()
        model.fit(x_poly, y_clean)
        
        # 预测
        y_pred = model.predict(x_poly)
        
        # 计算统计量
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        
        # 生成平滑曲线
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
        x_range_poly = poly_features.transform(x_range)
        y_range_pred = model.predict(x_range_poly)
        
        # 构建方程字符串
        coeffs = model.coef_
        equation_parts = []
        for i, coef in enumerate(coeffs[1:], 1):  # 跳过截距
            if abs(coef) > 1e-10:
                sign = '+' if coef > 0 else ''
                if i == 1:
                    equation_parts.append(f'{sign}{coef:.4f}x')
                else:
                    equation_parts.append(f'{sign}{coef:.4f}x^{i}')
        
        equation = f'y = {model.intercept_:.4f} ' + ' '.join(equation_parts)
        
        return {
            'degree': int(degree),
            'coefficients': coeffs.tolist(),
            'intercept': float(model.intercept_),
            'r_squared': float(r2),
            'rmse': float(rmse),
            'n_observations': int(len(x_clean)),
            'equation': equation,
            'fitted_values': y_pred.tolist(),
            'prediction_range': {
                'x': x_range.flatten().tolist(),
                'y': y_range_pred.tolist()
            }
        }


class HypothesisTesting:
    """假设检验"""
    
    @staticmethod
    def t_test_independent(
        group1: np.ndarray, 
        group2: np.ndarray,
        equal_var: bool = True
    ) -> Dict[str, Any]:
        """
        独立样本t检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            equal_var: 是否假设方差相等
            
        Returns:
            检验结果字典
        """
        g1_clean = group1[~np.isnan(group1)]
        g2_clean = group2[~np.isnan(group2)]
        
        if len(g1_clean) < 2 or len(g2_clean) < 2:
            return {'error': '样本量不足'}
        
        # t检验
        t_stat, p_value = stats.ttest_ind(g1_clean, g2_clean, equal_var=equal_var)
        
        # 效应量 (Cohen's d)
        pooled_std = np.sqrt(((len(g1_clean)-1)*np.var(g1_clean, ddof=1) + 
                               (len(g2_clean)-1)*np.var(g2_clean, ddof=1)) / 
                              (len(g1_clean)+len(g2_clean)-2))
        cohens_d = (np.mean(g1_clean) - np.mean(g2_clean)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'group1_mean': float(np.mean(g1_clean)),
            'group2_mean': float(np.mean(g2_clean)),
            'group1_std': float(np.std(g1_clean, ddof=1)),
            'group2_std': float(np.std(g2_clean, ddof=1)),
            'group1_n': int(len(g1_clean)),
            'group2_n': int(len(g2_clean)),
            'significant': bool(p_value < 0.05)
        }
    
    @staticmethod
    def normality_test(data: np.ndarray) -> Dict[str, Any]:
        """
        正态性检验 (Shapiro-Wilk)
        
        Args:
            data: 数据数组
            
        Returns:
            检验结果字典
        """
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return {'error': '样本量不足'}
        
        if len(data_clean) > 5000:
            # 对于大样本，使用K-S检验
            stat, p_value = stats.kstest(data_clean, 'norm', 
                                         args=(np.mean(data_clean), np.std(data_clean, ddof=1)))
            test_name = 'Kolmogorov-Smirnov'
        else:
            # Shapiro-Wilk检验
            stat, p_value = stats.shapiro(data_clean)
            test_name = 'Shapiro-Wilk'
        
        return {
            'test': test_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > 0.05),
            'n': int(len(data_clean))
        }


# 导出的主要接口函数
def analyze_descriptive_stats(data: Dict[str, List[float]]) -> Dict[str, Any]:
    """描述性统计分析接口"""
    df = pd.DataFrame(data)
    
    results = {}
    for col in df.columns:
        results[col] = DescriptiveStatistics.calculate_basic_stats(df[col])
    
    summary_table = DescriptiveStatistics.calculate_summary_table(df)
    
    return {
        'detailed_stats': results,
        'summary_table': summary_table.to_dict(orient='records')
    }


def analyze_correlation(
    data: Dict[str, List[float]], 
    method: str = 'pearson'
) -> Dict[str, Any]:
    """相关性分析接口"""
    df = pd.DataFrame(data)
    
    corr_matrix, pvalue_matrix = CorrelationAnalysis.calculate_correlation_matrix(
        df, method=method
    )
    
    significant_pairs = CorrelationAnalysis.get_significant_correlations(
        corr_matrix, pvalue_matrix
    )
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'p_value_matrix': pvalue_matrix.to_dict(),
        'significant_pairs': significant_pairs,
        'method': method
    }


def analyze_regression(
    x: List[float], 
    y: List[float],
    regression_type: str = 'linear',
    polynomial_degree: int = 2
) -> Dict[str, Any]:
    """回归分析接口"""
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    if regression_type == 'linear':
        result = RegressionAnalysis.linear_regression(x_arr, y_arr)
    elif regression_type == 'polynomial':
        result = RegressionAnalysis.polynomial_regression(x_arr, y_arr, degree=polynomial_degree)
    else:
        return {'error': f'不支持的回归类型: {regression_type}'}
    
    return result
