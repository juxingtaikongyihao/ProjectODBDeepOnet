from pathlib import Path
from typing import Dict
import numpy as np

"""
output.py

Unified output management and VTK writer.

Provides:
- project_root()
- default_output_root()
- prepare_output_structure(root=None) -> dict of categorized dirs
- write_point_vtk(filename, points, point_scalars)
"""

def project_root() -> Path:
    # assumes package layout: repo_root/deeponet/...
    return Path(__file__).resolve().parent.parent

def default_output_root() -> Path:
    return project_root() / "output"

def prepare_output_structure(root: str | Path | None = None) -> Dict[str, Path]:
    """
    Create and return categorized output directories.
    Keys:
      - root
      - models
      - predictions
      - predictions_baseline
      - predictions_stacked
      - history
      - analysis
      - logs
      - prepared_data
    """
    root = Path(root) if root is not None else default_output_root()
    dirs = {
        "root": root,
        "models": root / "models",
        "predictions": root / "predictions",
        "predictions_baseline": root / "predictions" / "baseline",
        "predictions_stacked": root / "predictions" / "stacked",
        "history": root / "history",
        "analysis": root / "analysis",
        "logs": root / "logs",
        "prepared_data": root / "prepared_data",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def write_point_vtk(filename: str, points: np.ndarray, point_scalars: dict):
    """
    Legacy ASCII VTK UnstructuredGrid with VERTEX cells.
    points: (N,3)
    point_scalars: dict name-> (N,) or (N,3) or (N,k)
    """
    N = points.shape[0]
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DeepONet point data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {N} float\n")
        for p in points:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
        total_size = N * 2
        f.write(f"\nCELLS {N} {total_size}\n")
        for i in range(N):
            f.write(f"1 {i}\n")
        f.write(f"\nCELL_TYPES {N}\n")
        for i in range(N):
            f.write("1\n")
        f.write(f"\nPOINT_DATA {N}\n")
        for name, arr in point_scalars.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for v in arr:
                    f.write(f"{v:.6e}\n")
            elif arr.ndim == 2 and arr.shape[1] == 3:
                f.write(f"VECTORS {name} float\n")
                for v in arr:
                    f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            else:
                arr_flat = arr.reshape(N, -1)
                for col in range(arr_flat.shape[1]):
                    nm = f"{name}_{col}"
                    f.write(f"SCALARS {nm} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for v in arr_flat[:, col]:
                        f.write(f"{v:.6e}\n")