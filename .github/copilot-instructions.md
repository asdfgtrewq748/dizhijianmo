## What This Repo Does
- 3D geological modeling from borehole CSVs using PyTorch Geometric GNNs; outputs voxel models, VTK/NPZ, stats, and Streamlit viz.
- Two pipelines: (1) lithology classification (per-sample nodes) and (2) layer-based cumulative modeling (per-layer thickness).

## Key Entry Points
- `main.py` CLI: `demo`, `webapp`, `train`, `run` (full pipeline), `layer` (layer-based), `model` (inference only). Example: `python main.py train --data ./data --model graphsage --epochs 300`.
- `app.py`: Streamlit UI (same as `python main.py webapp`).

## Configuration
- Central config in `configs/config.py`: data (`graph_type`, `k_neighbors`), model (`model_type`, `num_layers`, `hidden_channels`), train (`loss_type`, `scheduler`, `focal_gamma`, augmentation, EMA), viz (`grid_resolution`). Defaults: `model_type='enhanced'`, `num_layers=3`, `hidden_channels=512`.
- Avoid hardcoding hyperparameters; read via `get_config` / `validate_config`.

## Data Contracts
- Borehole CSVs must include `x,y,z,lithology`; optional features (porosity, permeability, density, etc.). Coordinate CSV supplies `borehole_id,x,y`.
- `BoreholeDataProcessor` auto-detects encodings (utf-8/gbk/gb2312) and standardizes lithology names; by default it merges all coal layers into a single label (`merge_coal=True`). Set `merge_coal=False` (or use `LayerDataProcessor`) if you need per-coal-layer separation; otherwise you will see only a few merged layers.
- Graph building supports `knn`/`radius`/`delaunay`; edge weights are distance-based and converted to exp(-d/mean_d). `k_neighbors` is capped at `N-1` to avoid empty graphs.

## Modeling Workflows
- **Classification (per sample)**: `BoreholeDataProcessor.process` → PyG `Data(x, edge_index, edge_weight, y, masks)`. Model via `get_model` (`gcn`, `graphsage`, `gat`, `geo3d`, `enhanced`). Trainer: `GeoModelTrainer` (focal loss default, class weights via `compute_class_weights`, optional MixUp/edge drop/noise, EMA). Saves to `output/models/best_model.pt`, predictions to `output/predictions.csv`.
- **Layer-based (thickness regression)**: `LayerDataProcessor` (no coal merge, `min_layer_occurrence`=1 in `build_layer_based_model`) infers ordered layers, fills missing thickness by IDW, builds KNN graph per borehole. `GNNThicknessPredictor` predicts per-layer thickness; fallback RBF/linear interpolation when `use_gnn=False`. Surfaces are accumulated from bottom up; exports VTK/NPZ and stats.

## Important Behaviors / Pitfalls
- Coal collapsing: default classification path merges all coal layers; this is the usual reason “only two layers/classes” appear. For full layer separation, disable merge or use the layer-based pipeline.
- Rare classes (<2 samples) trigger non-stratified splits and warnings; expect unstable metrics if your data are that sparse.
- Feature auto-selection: numerical columns not in `{x,y,z,lithology,...}` with >10% non-null are used; engineered features add centroid distance, z-bins, log(thickness), and normalized `layer_order` when present.
- Masks: `train/val/test` masks created by stratified split when possible; early stopping/patience controlled in trainer `train` args.

## Modeling & Viz Outputs
- Full pipeline (`main.py run`/`demo`) also builds voxel models and saves stats + predictions; layer-based pipeline writes `layer_model.vtk`, `layer_model.npz`, `layer_model_stats.csv`, optional `thickness_model.pt`.

## Conventions
- Type hints throughout; use `logging` in core modules. Path handling via `os.path.join`/`pathlib`. Keep ASCII in code/comments.
