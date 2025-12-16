# backend/batch_operations.py
"""
批量操作模块
支持批量导入、批量修改、批量导出等功能
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class OperationStatus(Enum):
    """操作状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchResult:
    """批量操作结果"""
    total: int
    success: int
    failed: int
    skipped: int
    errors: List[Dict[str, Any]]
    details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": round(self.success / self.total * 100, 2) if self.total > 0 else 0,
            "errors": self.errors,
            "details": self.details
        }


class BatchOperator:
    """批量操作器"""

    def __init__(self, max_concurrent: int = 5):
        """
        Args:
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent

    async def execute_batch(self,
                           items: List[Any],
                           operation: Callable,
                           progress_callback: Optional[Callable] = None) -> BatchResult:
        """
        执行批量操作

        Args:
            items: 要处理的项目列表
            operation: 操作函数
            progress_callback: 进度回调函数

        Returns:
            批量操作结果
        """
        total = len(items)
        results = []
        errors = []

        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_item(index: int, item: Any):
            """处理单个项目"""
            async with semaphore:
                try:
                    result = await operation(item)
                    results.append({
                        "index": index,
                        "status": OperationStatus.SUCCESS.value,
                        "result": result
                    })

                    if progress_callback:
                        await progress_callback(index + 1, total, OperationStatus.SUCCESS)

                except Exception as e:
                    error_info = {
                        "index": index,
                        "item": str(item)[:100],  # 限制长度
                        "error": str(e)
                    }
                    errors.append(error_info)
                    results.append({
                        "index": index,
                        "status": OperationStatus.FAILED.value,
                        "error": str(e)
                    })

                    if progress_callback:
                        await progress_callback(index + 1, total, OperationStatus.FAILED)

        # 创建所有任务
        tasks = [process_item(i, item) for i, item in enumerate(items)]

        # 并发执行
        await asyncio.gather(*tasks)

        # 统计结果
        success_count = sum(1 for r in results if r["status"] == OperationStatus.SUCCESS.value)
        failed_count = sum(1 for r in results if r["status"] == OperationStatus.FAILED.value)

        return BatchResult(
            total=total,
            success=success_count,
            failed=failed_count,
            skipped=0,
            errors=errors,
            details=results
        )

    def batch_import_files(self,
                          file_list: List[str],
                          import_function: Callable,
                          retry_on_failure: bool = True) -> BatchResult:
        """
        批量导入文件

        Args:
            file_list: 文件路径列表
            import_function: 导入函数
            retry_on_failure: 失败时是否重试

        Returns:
            批量操作结果
        """
        results = []
        errors = []
        success_count = 0
        failed_count = 0
        skipped_count = 0

        for i, file_path in enumerate(file_list):
            try:
                # 尝试导入
                result = import_function(file_path)

                results.append({
                    "index": i,
                    "file": file_path,
                    "status": OperationStatus.SUCCESS.value,
                    "result": result
                })
                success_count += 1

            except Exception as e:
                error_msg = str(e)

                # 如果启用重试
                if retry_on_failure:
                    try:
                        result = import_function(file_path)
                        results.append({
                            "index": i,
                            "file": file_path,
                            "status": OperationStatus.SUCCESS.value,
                            "result": result,
                            "retried": True
                        })
                        success_count += 1
                        continue
                    except Exception as retry_error:
                        error_msg = f"重试后仍失败: {retry_error}"

                # 记录失败
                errors.append({
                    "index": i,
                    "file": file_path,
                    "error": error_msg
                })
                results.append({
                    "index": i,
                    "file": file_path,
                    "status": OperationStatus.FAILED.value,
                    "error": error_msg
                })
                failed_count += 1

        return BatchResult(
            total=len(file_list),
            success=success_count,
            failed=failed_count,
            skipped=skipped_count,
            errors=errors,
            details=results
        )

    def batch_update_records(self,
                            records: List[Dict],
                            update_fields: Dict[str, Any],
                            condition: Optional[Callable] = None) -> BatchResult:
        """
        批量更新记录

        Args:
            records: 记录列表
            update_fields: 要更新的字段和值
            condition: 更新条件函数 (返回True表示更新)

        Returns:
            批量操作结果
        """
        results = []
        errors = []
        success_count = 0
        skipped_count = 0

        for i, record in enumerate(records):
            try:
                # 检查条件
                if condition and not condition(record):
                    results.append({
                        "index": i,
                        "status": OperationStatus.SKIPPED.value,
                        "reason": "不满足更新条件"
                    })
                    skipped_count += 1
                    continue

                # 更新记录
                for field, value in update_fields.items():
                    record[field] = value

                results.append({
                    "index": i,
                    "status": OperationStatus.SUCCESS.value,
                    "updated_fields": list(update_fields.keys())
                })
                success_count += 1

            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e)
                })
                results.append({
                    "index": i,
                    "status": OperationStatus.FAILED.value,
                    "error": str(e)
                })

        return BatchResult(
            total=len(records),
            success=success_count,
            failed=len(errors),
            skipped=skipped_count,
            errors=errors,
            details=results
        )

    def batch_export_data(self,
                         data_groups: Dict[str, pd.DataFrame],
                         export_function: Callable,
                         format: str = "csv") -> BatchResult:
        """
        批量导出数据

        Args:
            data_groups: 数据分组字典 {组名: DataFrame}
            export_function: 导出函数
            format: 导出格式

        Returns:
            批量操作结果
        """
        results = []
        errors = []
        success_count = 0

        for group_name, df in data_groups.items():
            try:
                # 导出数据
                output_path = export_function(df, group_name, format)

                results.append({
                    "group": group_name,
                    "status": OperationStatus.SUCCESS.value,
                    "output_path": output_path,
                    "row_count": len(df)
                })
                success_count += 1

            except Exception as e:
                errors.append({
                    "group": group_name,
                    "error": str(e)
                })
                results.append({
                    "group": group_name,
                    "status": OperationStatus.FAILED.value,
                    "error": str(e)
                })

        return BatchResult(
            total=len(data_groups),
            success=success_count,
            failed=len(errors),
            skipped=0,
            errors=errors,
            details=results
        )


class DataGrouper:
    """数据分组器"""

    @staticmethod
    def group_by_column(df: pd.DataFrame, column: str) -> Dict[str, pd.DataFrame]:
        """
        按列分组

        Args:
            df: 数据框
            column: 分组列

        Returns:
            分组字典
        """
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在")

        groups = {}
        for value in df[column].unique():
            group_df = df[df[column] == value]
            groups[str(value)] = group_df

        return groups

    @staticmethod
    def group_by_size(df: pd.DataFrame, size: int) -> Dict[str, pd.DataFrame]:
        """
        按大小分组

        Args:
            df: 数据框
            size: 每组大小

        Returns:
            分组字典
        """
        groups = {}
        total_rows = len(df)

        for i in range(0, total_rows, size):
            group_name = f"group_{i//size + 1}"
            groups[group_name] = df.iloc[i:i+size]

        return groups

    @staticmethod
    def group_by_custom(df: pd.DataFrame,
                       grouping_function: Callable) -> Dict[str, pd.DataFrame]:
        """
        自定义分组

        Args:
            df: 数据框
            grouping_function: 分组函数 (row -> group_name)

        Returns:
            分组字典
        """
        groups = {}

        for idx, row in df.iterrows():
            group_name = grouping_function(row)

            if group_name not in groups:
                groups[group_name] = []

            groups[group_name].append(row)

        # 转换为DataFrame
        for group_name in groups:
            groups[group_name] = pd.DataFrame(groups[group_name])

        return groups


# 便捷函数
def batch_import(file_list: List[str],
                import_function: Callable,
                max_concurrent: int = 5) -> BatchResult:
    """批量导入的便捷函数"""
    operator = BatchOperator(max_concurrent=max_concurrent)
    return operator.batch_import_files(file_list, import_function)


def batch_update(records: List[Dict],
                update_fields: Dict[str, Any],
                condition: Optional[Callable] = None) -> BatchResult:
    """批量更新的便捷函数"""
    operator = BatchOperator()
    return operator.batch_update_records(records, update_fields, condition)


def batch_export(data_groups: Dict[str, pd.DataFrame],
                export_function: Callable,
                format: str = "csv") -> BatchResult:
    """批量导出的便捷函数"""
    operator = BatchOperator()
    return operator.batch_export_data(data_groups, export_function, format)
