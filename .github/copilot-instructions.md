# GitHub Copilot Instructions for GNN Geological Modeling

This project implements a 3D geological modeling system using Graph Neural Networks (GNNs) and PyTorch Geometric.

## Project Architecture

- **Core Logic**: `src/` contains the primary components:
  - `data_loader.py`: Converts borehole CSV data into graph structures (nodes=sample points, edges=spatial proximity).
  - `models.py`: Defines GNN architectures (GCN, GraphSAGE, GAT, Geo3D).
  - `trainer.py`: Manages the training loop, evaluation, and model persistence.
- **Configuration**: `configs/config.py` centralizes all hyperparameters for data, models, and training.
- **Entry Points**:
  - `main.py`: CLI for training, demo, and evaluation.
  - `app.py`: Streamlit web application for interactive visualization.

## Critical Workflows

### Data Processing
- **Input Format**: The system expects specific CSV formats (Min Dong mining area style).
  - **Coordinate File**: Contains borehole IDs and X, Y coordinates.
  - **Borehole Files**: Contain depth-based measurements (thickness, lithology, etc.).
- **Graph Construction**: Use `BoreholeDataProcessor` in `src/data_loader.py`. It handles:
  - Merging coordinates with borehole data.
  - Constructing graphs using KNN, Radius, or Delaunay triangulation (`config.DATA_CONFIG['graph_type']`).
  - Handling text encoding (`utf-8`, `gbk`, etc.) automatically.

### Training & Evaluation
- **Configuration**: Always check and modify `configs/config.py` before training. Do not hardcode hyperparameters.
- **Loss Functions**: The project uses `FocalLoss` (default) or `LabelSmoothingLoss` to handle class imbalance in lithology data.
- **Command**: Run training via `python main.py train --data <path> --model <type>`.

### Visualization
- Use `app.py` for 3D visualization of the geological model.
- Ensure `plotly` is used for 3D scatter and mesh plots.

## Coding Conventions

- **PyTorch Geometric**: Use `torch_geometric.data.Data` objects. Ensure `x` (features), `edge_index` (connectivity), and `y` (labels) are correctly shaped.
- **Path Handling**: Use `os.path.join` or `pathlib` for cross-platform compatibility (Windows/Linux).
- **Type Hinting**: Use Python type hints (`typing.List`, `typing.Optional`, etc.) for all function signatures.
- **Logging**: Use the `logging` module instead of `print` for status updates in core modules.

## Common Patterns

- **Model Factory**: Use `src.models.get_model(name, ...)` to instantiate models.
- **Data Loading**:
  ```python
  from src.data_loader import BoreholeDataProcessor
  processor = BoreholeDataProcessor(k_neighbors=30)
  # Load coordinates first, then borehole data
  ```
- **Device Management**: Use `device = 'cuda' if torch.cuda.is_available() else 'cpu'` but prefer the `auto` setting in `GeoModelTrainer`.

## Dependencies
- **Core**: `torch`, `torch_geometric`
- **Data**: `pandas`, `numpy`, `scikit-learn`
- **Viz**: `streamlit`, `plotly`
