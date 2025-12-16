# backend/visualization_enhanced.py
"""
数据可视化增强模块
提供高级图表生成功能
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class ChartGenerator:
    """图表生成器"""

    @staticmethod
    def generate_rose_chart_data(df: pd.DataFrame,
                                 category_col: str,
                                 value_col: str) -> Dict[str, Any]:
        """
        生成玫瑰图数据(南丁格尔玫瑰图)

        Args:
            df: 数据框
            category_col: 分类列
            value_col: 数值列

        Returns:
            ECharts配置
        """
        # 统计各类别数量
        stats = df.groupby(category_col)[value_col].agg(['count', 'sum']).reset_index()

        categories = stats[category_col].tolist()
        values = stats['count'].tolist()

        data = [
            {
                "value": int(value),
                "name": str(category)
            }
            for category, value in zip(categories, values)
        ]

        chart_config = {
            "title": {
                "text": f"{category_col}分布(玫瑰图)",
                "left": "center"
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{b}: {c} ({d}%)"
            },
            "series": [{
                "type": "pie",
                "radius": ["30%", "70%"],
                "roseType": "area",
                "itemStyle": {
                    "borderRadius": 8
                },
                "data": data,
                "label": {
                    "show": True,
                    "formatter": "{b}: {d}%"
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }]
        }

        return chart_config

    @staticmethod
    def generate_sankey_diagram_data(df: pd.DataFrame,
                                     source_col: str,
                                     target_col: str,
                                     value_col: Optional[str] = None) -> Dict[str, Any]:
        """
        生成桑基图数据

        Args:
            df: 数据框
            source_col: 源节点列
            target_col: 目标节点列
            value_col: 值列(可选,默认为计数)

        Returns:
            ECharts配置
        """
        # 如果没有指定值列,使用计数
        if value_col:
            flow_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
        else:
            flow_data = df.groupby([source_col, target_col]).size().reset_index(name='count')
            value_col = 'count'

        # 提取所有唯一节点
        nodes = set(df[source_col].unique()) | set(df[target_col].unique())
        nodes_data = [{"name": str(node)} for node in nodes]

        # 生成链接数据
        links_data = [
            {
                "source": str(row[source_col]),
                "target": str(row[target_col]),
                "value": float(row[value_col])
            }
            for _, row in flow_data.iterrows()
        ]

        chart_config = {
            "title": {
                "text": f"{source_col} → {target_col} 流向图",
                "left": "center"
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{a} → {b}: {c}"
            },
            "series": [{
                "type": "sankey",
                "layout": "none",
                "emphasis": {
                    "focus": "adjacency"
                },
                "data": nodes_data,
                "links": links_data,
                "label": {
                    "show": True,
                    "position": "right"
                },
                "lineStyle": {
                    "color": "gradient",
                    "curveness": 0.5
                }
            }]
        }

        return chart_config

    @staticmethod
    def generate_radar_chart_data(df: pd.DataFrame,
                                  metrics: List[str],
                                  name_col: str) -> Dict[str, Any]:
        """
        生成雷达图数据(多维度对比)

        Args:
            df: 数据框
            metrics: 指标列表
            name_col: 名称列

        Returns:
            ECharts配置
        """
        # 检查列是否存在
        for metric in metrics:
            if metric not in df.columns:
                raise ValueError(f"指标列 '{metric}' 不存在")

        if name_col not in df.columns:
            raise ValueError(f"名称列 '{name_col}' 不存在")

        # 标准化数据到0-100范围
        normalized_df = df.copy()
        for metric in metrics:
            col = normalized_df[metric]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                normalized_df[metric] = (col - min_val) / (max_val - min_val) * 100
            else:
                normalized_df[metric] = 50  # 如果所有值相同,设为50

        # 生成指标配置
        indicator = [
            {
                "name": metric,
                "max": 100
            }
            for metric in metrics
        ]

        # 生成系列数据
        series_data = []
        for _, row in normalized_df.iterrows():
            values = [float(row[metric]) for metric in metrics]
            series_data.append({
                "name": str(row[name_col]),
                "value": values
            })

        chart_config = {
            "title": {
                "text": "多维度参数对比(雷达图)",
                "left": "center"
            },
            "tooltip": {
                "trigger": "item"
            },
            "legend": {
                "orient": "vertical",
                "left": "left",
                "data": [str(row[name_col]) for _, row in df.iterrows()]
            },
            "radar": {
                "indicator": indicator,
                "shape": "circle",
                "splitNumber": 5,
                "name": {
                    "textStyle": {
                        "color": "#333"
                    }
                },
                "splitLine": {
                    "lineStyle": {
                        "color": [
                            'rgba(0, 0, 0, 0.1)',
                            'rgba(0, 0, 0, 0.1)',
                            'rgba(0, 0, 0, 0.1)',
                            'rgba(0, 0, 0, 0.1)',
                            'rgba(0, 0, 0, 0.1)'
                        ]
                    }
                },
                "splitArea": {
                    "show": True,
                    "areaStyle": {
                        "color": ['rgba(0, 0, 0, 0.05)', 'rgba(0, 0, 0, 0.025)']
                    }
                }
            },
            "series": [{
                "type": "radar",
                "data": series_data,
                "emphasis": {
                    "lineStyle": {
                        "width": 4
                    }
                }
            }]
        }

        return chart_config

    @staticmethod
    def generate_heatmap_data(df: pd.DataFrame,
                             x_col: str,
                             y_col: str,
                             value_col: str) -> Dict[str, Any]:
        """
        生成热力图数据

        Args:
            df: 数据框
            x_col: X轴列
            y_col: Y轴列
            value_col: 值列

        Returns:
            ECharts配置
        """
        # 透视表
        pivot = df.pivot_table(
            index=y_col,
            columns=x_col,
            values=value_col,
            aggfunc='mean'
        )

        x_categories = pivot.columns.tolist()
        y_categories = pivot.index.tolist()

        # 生成数据
        data = []
        for i, y in enumerate(y_categories):
            for j, x in enumerate(x_categories):
                value = pivot.loc[y, x]
                if pd.notna(value):
                    data.append([j, i, float(value)])

        chart_config = {
            "title": {
                "text": f"{value_col}热力图",
                "left": "center"
            },
            "tooltip": {
                "position": "top",
                "formatter": lambda params: f"{x_categories[params['data'][0]]}, {y_categories[params['data'][1]]}: {params['data'][2]:.2f}"
            },
            "grid": {
                "height": "50%",
                "top": "10%"
            },
            "xAxis": {
                "type": "category",
                "data": x_categories,
                "splitArea": {
                    "show": True
                }
            },
            "yAxis": {
                "type": "category",
                "data": y_categories,
                "splitArea": {
                    "show": True
                }
            },
            "visualMap": {
                "min": float(pivot.min().min()) if not pivot.empty else 0,
                "max": float(pivot.max().max()) if not pivot.empty else 100,
                "calculable": True,
                "orient": "horizontal",
                "left": "center",
                "bottom": "5%"
            },
            "series": [{
                "type": "heatmap",
                "data": data,
                "label": {
                    "show": True
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }]
        }

        return chart_config

    @staticmethod
    def generate_comparison_chart_data(dataframes: Dict[str, pd.DataFrame],
                                      metric_col: str,
                                      category_col: str) -> Dict[str, Any]:
        """
        生成对比图数据(多个钻孔/岩层对比)

        Args:
            dataframes: 数据框字典 {名称: DataFrame}
            metric_col: 指标列
            category_col: 分类列

        Returns:
            ECharts配置
        """
        # 提取所有分类
        all_categories = set()
        for df in dataframes.values():
            if category_col in df.columns:
                all_categories.update(df[category_col].unique())

        categories = sorted(list(all_categories))

        # 生成系列数据
        series = []
        for name, df in dataframes.items():
            if metric_col not in df.columns or category_col not in df.columns:
                continue

            # 按分类聚合
            agg_data = df.groupby(category_col)[metric_col].mean().to_dict()

            values = [float(agg_data.get(cat, 0)) for cat in categories]

            series.append({
                "name": name,
                "type": "bar",
                "data": values,
                "emphasis": {
                    "focus": "series"
                },
                "animationDelay": lambda idx: idx * 10
            })

        chart_config = {
            "title": {
                "text": f"{metric_col}对比分析",
                "left": "center"
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {
                    "type": "shadow"
                }
            },
            "legend": {
                "data": list(dataframes.keys()),
                "top": "bottom"
            },
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "10%",
                "containLabel": True
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisTick": {
                    "alignWithLabel": True
                }
            },
            "yAxis": {
                "type": "value"
            },
            "series": series
        }

        return chart_config


# 便捷函数
def create_rose_chart(df: pd.DataFrame, category_col: str, value_col: str) -> Dict[str, Any]:
    """创建玫瑰图的便捷函数"""
    generator = ChartGenerator()
    return generator.generate_rose_chart_data(df, category_col, value_col)


def create_sankey_diagram(df: pd.DataFrame, source_col: str, target_col: str,
                         value_col: Optional[str] = None) -> Dict[str, Any]:
    """创建桑基图的便捷函数"""
    generator = ChartGenerator()
    return generator.generate_sankey_diagram_data(df, source_col, target_col, value_col)


def create_radar_chart(df: pd.DataFrame, metrics: List[str], name_col: str) -> Dict[str, Any]:
    """创建雷达图的便捷函数"""
    generator = ChartGenerator()
    return generator.generate_radar_chart_data(df, metrics, name_col)
