#!/usr/bin/env python3
"""
run_stacked.py

Stacked-model training & prediction entry. Saves artifacts into project_root/output (organized).
This file is adjusted so it can be executed directly (python deeponet/run_stacked.py)
or as a module (python -m deeponet.run_stacked).
"""
import argparse
import os
from pathlib import Path
import sys

# If executed directly (not as a package), ensure project root is on sys.path and set package
if __name__ == "__main__" and __package__ is None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    __package__ = "deeponet"

# Use relative imports
from .data import load_data
from . import train as train_module
from .output import prepare_output_structure, project_root

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
    parser = argparse.ArgumentParser(description="Train stacked DeepONet and save outputs to output/ ...")
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

    # call shared train_and_predict from train module; instruct save_dir into predictions_stacked folder
    save_dir = str(out_dirs["predictions_stacked"])
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
            model_path = out_dirs["models"] / "stacked_final.pth"
            torch.save(model.state_dict(), str(model_path))
            print("Saved model to:", model_path)
    except Exception as e:
        print("Warning: failed to save final model to output models dir:", e)

    preds_path = result.get("predictions_path", None)
    if preds_path:
        print("Predictions saved to:", preds_path)
    else:
        print("No predictions_path returned by train_and_predict; check train module. If predictions were saved to save_dir, look under:", save_dir)

    print("Stacked run finished. Output root:", out_dirs["root"])

if __name__ == "__main__":
    main()