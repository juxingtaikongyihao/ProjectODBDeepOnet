#!/usr/bin/env python3
"""
run_baseline.py

Blend of safe imports and file-based fallback imports so you can run:
  python deeponet/run_baseline.py ...
or
  python -m deeponet.run_baseline ...
"""
import argparse
from pathlib import Path
import sys

def _load_module_from_path(module_name: str, path: Path):
    import importlib.util
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod

# Try normal package imports first; if they fail, fall back to loading files directly.
try:
    from deeponet.data import load_data
    from deeponet import train as train_module
    from deeponet.output import prepare_output_structure
except Exception:
    # compute repo root and deeponet dir
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent
    deeponet_dir = repo_root / "deeponet"
    # load modules by path
    data_mod = _load_module_from_path("deeponet.data", deeponet_dir / "data.py")
    train_module = _load_module_from_path("deeponet.train", deeponet_dir / "train.py")
    output_mod = _load_module_from_path("deeponet.output", deeponet_dir / "output.py")
    # expose needed names
    load_data = getattr(data_mod, "load_data")
    prepare_output_structure = getattr(output_mod, "prepare_output_structure")

def check_torch():
    try:
        import torch
        v = torch.__version__
        cuda_v = getattr(torch.version, "cuda", None)
        print(f"torch.__version__ = {v}, torch.version.cuda = {cuda_v}, cuda available = {torch.cuda.is_available()}")
        if not v.startswith("2.9"):
            print("WARNING: Detected torch version not starting with 2.9 — please verify compatibility with PyTorch 2.9.0.")
    except Exception as e:
        print("WARNING: Could not import torch to check version:", e)

def parse_args(argv=None):
    import torch
    parser = argparse.ArgumentParser(description="Train baseline DeepONet and save outputs to output/ ...")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to train_data (auto-detect project_root/train_data if not provided)")
    parser.add_argument("--output-dir", type=str, default=None, help="Root output directory (default project_root/output)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--save-plot", action="store_true", dest="save_plot")
    parser.add_argument("--export-vtk", action="store_true", dest="export_vtk")
    parser.add_argument("--vtk-samples", type=str, default="0")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    check_torch()

    # auto-detect data_dir
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        pr = script_dir.parent
        candidate = pr / "train_data"
        data_dir = str(candidate if candidate.exists() else (Path.cwd() / "train_data"))
    else:
        data_dir = args.data_dir

    # prepare output structure
    out_root = args.output_dir if args.output_dir is not None else None
    out_dirs = prepare_output_structure(out_root)

    print("Using data_dir:", data_dir)
    print("Saving outputs under:", out_dirs["root"])

    # call shared train_and_predict from train module; instruct save_dir into predictions baseline folder
    save_dir = str(out_dirs["predictions_baseline"])
    result = train_module.train_and_predict(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rank=args.rank,
        weight_decay=args.weight_decay,
        device=args.device,
        save_plot=args.save_plot,
        export_vtk=args.export_vtk,
        vtk_samples=args.vtk_samples,
        save_dir=save_dir,
        dropout=args.dropout
    )

    # try to save model into output models dir if returned
    try:
        import torch
        model = result.get("model", None)
        if model is not None:
            model_path = out_dirs["models"] / "baseline_final.pth"
            torch.save(model.state_dict(), str(model_path))
            print("Saved model to:", model_path)
    except Exception as e:
        print("Warning: failed to save final model to output models dir:", e)

    preds_path = result.get("predictions_path", None)
    if preds_path:
        print("Predictions saved to:", preds_path)
    else:
        print("No predictions_path returned by train_and_predict; check train module. If predictions were saved to save_dir, look under:", save_dir)

    print("Baseline run finished. Output root:", out_dirs["root"])

if __name__ == "__main__":
    main()