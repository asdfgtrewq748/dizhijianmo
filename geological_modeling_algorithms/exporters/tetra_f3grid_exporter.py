"""T4 (tetra) FLAC3D grid exporter.

该导出器基于规则网格 (XI, YI) 以及逐层 BlockModel 数据, 将每个 hexa “柱”
拆分为 6 个四面体 (Z T4). 通过结构化拓扑可以适应任意起伏/弯曲地形, 避免
B8 网格在几何扭曲时产生的负体积问题。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import fabs
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from coal_seam_blocks.modeling import BlockModel
from .base_exporter import BaseExporter


@dataclass
class GridPoint:
    gid: int
    x: float
    y: float
    z: float


@dataclass
class TetZone:
    zid: int
    gp_ids: Tuple[int, int, int, int]
    group: str


def _signed_tet_volume(p0: np.ndarray, p1: np.ndarray,
                       p2: np.ndarray, p3: np.ndarray) -> float:
    """计算四面体有向体积 (>0 表示右手系)。"""
    mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    return float(np.linalg.det(mat) / 6.0)


class TetraF3GridExporter(BaseExporter):
    """将 BlockModel 栈导出为 FLAC3D 原生 T4 网格 (.f3grid)."""

    def __init__(self) -> None:
        self.gridpoints: List[GridPoint] = []
        self.tet_zones: List[TetZone] = []
        self.layer_zone_ids: Dict[str, List[int]] = {}
        self._coords: Dict[int, np.ndarray] = {}
        self._min_tet_volume: float = 1e-6
        self._offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------
    def export(self, data: Any, output_path: str,
               options: Optional[Dict[str, Any]] = None) -> str:  # type: ignore[override]
        options = options or {}

        block_models = data.get("block_models") if isinstance(data, dict) else None
        XI = data.get("grid_x") if isinstance(data, dict) else None
        YI = data.get("grid_y") if isinstance(data, dict) else None
        if not block_models:
            raise ValueError("block_models 不能为空")
        if XI is None or YI is None:
            raise ValueError("grid_x/grid_y 不能为空")

        block_models = list(block_models)
        XI = np.asarray(XI, dtype=float)
        YI = np.asarray(YI, dtype=float)
        if XI.shape != YI.shape:
            raise ValueError("grid_x 与 grid_y 形状不一致")

        first_shape = block_models[0].top_surface.shape
        if XI.shape != first_shape:
            raise ValueError("grid_x/grid_y 形状需与 BlockModel 网格一致")

        self._reset()
        self._min_tet_volume = float(options.get("min_tet_volume", 1e-6))
        normalize_coords = bool(options.get("normalize_coords"))
        raw_offset = options.get("coordinate_offset") or options.get("coord_offset")
        if raw_offset is not None:
            self._offset = self._validate_offset(raw_offset)
        elif normalize_coords:
            self._offset = self._compute_auto_offset(XI, YI, block_models)
        else:
            self._offset = (0.0, 0.0, 0.0)

        interfaces = self._build_interfaces(block_models)
        node_ids = self._build_gridpoints(interfaces, XI, YI)
        self._build_tet_zones(block_models, interfaces, node_ids)
        self._write_f3grid(output_path)

        print(f"[TetraF3GridExporter] GridPoints: {len(self.gridpoints)} | "
              f"Zones(T4): {len(self.tet_zones)} | Groups: {len(self.layer_zone_ids)}")
        print(f"[TetraF3GridExporter] 输出: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # preparation helpers
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self.gridpoints = []
        self.tet_zones = []
        self.layer_zone_ids = {}
        self._coords = {}

    @staticmethod
    def _validate_offset(values: Sequence[float]) -> Tuple[float, float, float]:
        if len(values) != 3:
            raise ValueError("coordinate_offset 必须包含 (x, y, z)")
        return tuple(float(v) for v in values)

    def _compute_auto_offset(self, XI: np.ndarray, YI: np.ndarray,
                              block_models: List[BlockModel]) -> Tuple[float, float, float]:
        xs = XI[np.isfinite(XI)]
        ys = YI[np.isfinite(YI)]
        zs = []
        for bm in block_models:
            zs.append(np.nanmin(bm.bottom_surface))
        if xs.size and ys.size and zs:
            return (float(np.median(xs)), float(np.median(ys)), float(np.min(zs)))
        return (0.0, 0.0, 0.0)

    def _build_interfaces(self, block_models: List[BlockModel]) -> np.ndarray:
        n_layers = len(block_models)
        ny, nx = block_models[0].top_surface.shape
        interfaces = np.zeros((n_layers + 1, ny, nx), dtype=float)

        interfaces[0] = np.asarray(block_models[0].bottom_surface, dtype=float)
        for idx, model in enumerate(block_models):
            interfaces[idx + 1] = np.asarray(model.top_surface, dtype=float)

        self._enforce_columnwise_monotonic(interfaces)
        return interfaces

    @staticmethod
    def _enforce_columnwise_monotonic(interfaces: np.ndarray, eps: float = 1e-3) -> None:
        n_interfaces, ny, nx = interfaces.shape
        for j in range(ny):
            for i in range(nx):
                for k in range(1, n_interfaces):
                    if interfaces[k, j, i] < interfaces[k - 1, j, i] + eps:
                        interfaces[k, j, i] = interfaces[k - 1, j, i] + eps

    def _build_gridpoints(self, interfaces: np.ndarray,
                          XI: np.ndarray, YI: np.ndarray) -> np.ndarray:
        n_interfaces, ny, nx = interfaces.shape
        node_ids = np.zeros((n_interfaces, ny, nx), dtype=int)
        gid = 1
        x_off, y_off, z_off = self._offset
        for k in range(n_interfaces):
            z_layer = interfaces[k]
            for j in range(ny):
                for i in range(nx):
                    x = float(XI[j, i]) - x_off
                    y = float(YI[j, i]) - y_off
                    z = float(z_layer[j, i]) - z_off
                    self.gridpoints.append(GridPoint(gid, x, y, z))
                    self._coords[gid] = np.array([x, y, z], dtype=float)
                    node_ids[k, j, i] = gid
                    gid += 1
        return node_ids

    def _build_tet_zones(self, block_models: List[BlockModel],
                         interfaces: np.ndarray, node_ids: np.ndarray) -> None:
        n_layers = len(block_models)
        _, ny, nx = interfaces.shape
        zid = 1
        for layer_idx in range(n_layers):
            layer_name = block_models[layer_idx].name
            if layer_name not in self.layer_zone_ids:
                self.layer_zone_ids[layer_name] = []

            for j in range(ny - 1):
                for i in range(nx - 1):
                    n0 = node_ids[layer_idx, j, i]
                    n1 = node_ids[layer_idx, j, i + 1]
                    n2 = node_ids[layer_idx, j + 1, i + 1]
                    n3 = node_ids[layer_idx, j + 1, i]
                    n4 = node_ids[layer_idx + 1, j, i]
                    n5 = node_ids[layer_idx + 1, j, i + 1]
                    n6 = node_ids[layer_idx + 1, j + 1, i + 1]
                    n7 = node_ids[layer_idx + 1, j + 1, i]
                    hex_nodes = (n0, n1, n2, n3, n4, n5, n6, n7)
                    zid = self._add_tets_for_hex(hex_nodes, layer_name, zid)

    def _add_tets_for_hex(self, hex_nodes: Tuple[int, int, int, int,
                                                 int, int, int, int],
                          layer_name: str, next_zone_id: int) -> int:
        patterns = (
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
            (0, 5, 1, 6),
        )
        removed = 0
        for a, b, c, d in patterns:
            ga, gb, gc, gd = hex_nodes[a], hex_nodes[b], hex_nodes[c], hex_nodes[d]
            p0, p1, p2, p3 = self._coords[ga], self._coords[gb], self._coords[gc], self._coords[gd]
            vol = _signed_tet_volume(p0, p1, p2, p3)
            if fabs(vol) < self._min_tet_volume:
                removed += 1
                continue
            if vol < 0.0:
                gb, gc = gc, gb  # 翻转顺序以保证正体积
            self.tet_zones.append(TetZone(next_zone_id, (ga, gb, gc, gd), layer_name))
            self.layer_zone_ids[layer_name].append(next_zone_id)
            next_zone_id += 1
        if removed:
            print(f"[TetraF3GridExporter] skip {removed} degenerate tets in layer {layer_name}")
        return next_zone_id

    # ------------------------------------------------------------------
    # file writer
    # ------------------------------------------------------------------
    def _sanitize_group_name(self, name: str) -> str:
        mapping = {
            "煤": "coal",
            "砂质泥岩": "sandy_mudstone",
            "炭质泥岩": "carbonaceous_mudstone",
            "高岭质泥岩": "kaolinite_mudstone",
            "高岭岩": "kaolinite_rock",
            "风化煤": "weathered_coal",
            "含砾": "conglomeratic",
            "泥岩": "mudstone",
            "砂岩": "sandstone",
        }
        sanitized = name or "group"
        for key, value in mapping.items():
            sanitized = sanitized.replace(key, value)
        sanitized = re.sub(r"[^0-9A-Za-z_ ]", "_", sanitized)
        sanitized = sanitized.strip() or "group"
        return sanitized

    def _write_f3grid(self, output_path: str) -> None:
        total_gp = len(self.gridpoints)
        total_zones = len(self.tet_zones)
        total_groups = len(self.layer_zone_ids)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("* ====================================\n")
            f.write("* FLAC3D Native Grid File (T4)\n")
            f.write("* Generated by CoalSeam3D System\n")
            f.write("* ====================================\n")
            f.write(f"* Total GridPoints: {total_gp}\n")
            f.write(f"* Total Zones: {total_zones}\n")
            f.write(f"* Total Groups: {total_groups}\n")
            f.write("* ====================================\n\n")

            f.write("* GRIDPOINTS\n")
            f.write("*   G <id> <x> <y> <z>\n")
            for gp in self.gridpoints:
                f.write(f"G {gp.gid} {gp.x:.6f} {gp.y:.6f} {gp.z:.6f}\n")
            f.write("\n")

            f.write("* ZONES (T4)\n")
            f.write("*   Z T4 <id> <gp0> <gp1> <gp2> <gp3>\n")
            for zone in self.tet_zones:
                g0, g1, g2, g3 = zone.gp_ids
                f.write(f"Z T4 {zone.zid} {g0} {g1} {g2} {g3}\n")
            f.write("\n")

            f.write("* ZONE GROUPS\n")
            f.write("*   ZGROUP 'name'\n")
            f.write("*   <zone_id> <zone_id> ...\n")
            for name, ids in self.layer_zone_ids.items():
                safe_name = self._sanitize_group_name(name)
                f.write(f"ZGROUP '{safe_name}'\n")
                line: List[str] = []
                for zid in ids:
                    line.append(str(zid))
                    if len(line) >= 20:
                        f.write(" ".join(line) + "\n")
                        line = []
                if line:
                    f.write(" ".join(line) + "\n")
                f.write("\n")

            f.write("* ====================================\n")
            f.write("* End of Grid File\n")
            f.write("* ====================================\n")

        print(f"[TetraF3GridExporter] 文件体积: {os.path.getsize(output_path) / 1024:.2f} KB")
