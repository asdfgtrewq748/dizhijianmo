import os
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_borehole_csv(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load a single borehole CSV file with robust encoding handling."""
    encodings_to_try = [encoding] if encoding else ["utf-8-sig", "gbk", "utf-8"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise RuntimeError(f"读取钻孔文件失败: {file_path} -> {e}") from e
    raise RuntimeError(f"无法解析钻孔文件编码: {file_path} -> {last_error}")


def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and handle variations."""
    if df is None or df.empty:
        return df

    rename_map = {
        "序号(从下到上)": "序号",
        "厚度/m": "厚度",
        "弹性模量/Gpa": "弹性模量",
        "弹性模量/GPa": "弹性模量",
        "容重/kN*m-3": "容重",
        "容重/kN·m-3": "容重",
        "抗拉强度/MPa": "抗拉强度",
    }
    df = df.rename(columns=rename_map)

    if "Unnamed: 6" in df.columns and "备注" not in df.columns:
        df = df.rename(columns={"Unnamed: 6": "备注"})

    return df


def aggregate_boreholes(borehole_files: List[str], coordinate_file: str, merge_column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate multiple borehole CSV files and merge with coordinates.

    Returns a tuple of (merged_strata_df, coordinate_df).
    """
    if not borehole_files:
        raise ValueError("未选择任何钻孔文件")
    if not coordinate_file:
        raise ValueError("请提供坐标文件")

    coords_df = load_borehole_csv(coordinate_file)

    merged_frames = []
    for file_path in borehole_files:
        df = load_borehole_csv(file_path)
        df = unify_columns(df)
        
        # 如果数据中没有钻孔名列，则从文件名提取
        if "钻孔名" not in df.columns:
            borehole_name = os.path.splitext(os.path.basename(file_path))[0]
            df.insert(0, "钻孔名", borehole_name)
        
        merged_frames.append(df)

    strata_df = pd.concat(merged_frames, ignore_index=True)

    if merge_column and merge_column in strata_df.columns and merge_column in coords_df.columns:
        merge_key = merge_column
    else:
        candidates = ["钻孔名", "borehole", "孔号", "hole_id", "ID", "id"]
        merge_key = next((col for col in candidates if col in strata_df.columns and col in coords_df.columns), None)

    if merge_key is None:
        raise RuntimeError("钻孔数据与坐标数据缺少共同字段，无法合并。请确保两者包含相同的钻孔标识列。")

    merged_df = pd.merge(strata_df, coords_df, on=merge_key, how="inner")

    if merged_df.empty:
        raise RuntimeError("合并后的数据为空，请检查钻孔文件和坐标文件的匹配关系。")

    return merged_df, coords_df


def get_seam_groups(merged_df: pd.DataFrame, seam_column: str) -> Dict[str, pd.DataFrame]:
    """Group the merged dataframe by seam name."""
    if seam_column not in merged_df.columns:
        raise ValueError(f"在合并数据中未找到煤层列: {seam_column}")
    grouped = {}
    for seam_name, group_df in merged_df.groupby(seam_column):
        if seam_name is None or str(seam_name).strip() == "":
            continue
        grouped[str(seam_name)] = group_df.copy()
    return grouped
