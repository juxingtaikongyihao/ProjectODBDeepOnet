#!/usr/bin/env python3
"""
analyze_results.py

训练后结果分析与可视化脚本（中文注释）。

增强点：
- 支持同时分析 baseline 与 stacked（--which both），并自动生成对比 CSV/图。
- 自动定位 data_dir/results_dir，支持 --out-dir 覆盖分析输出目录，或使用 --output-dir 指向 project_root/output（默认）。
- 保持向后兼容的单文件分析路径。

用法示例：
python analyze_results.py --data-dir D:/Project/ProjectODBDeepOnet/train_data \
    --results-dir D:/Project/ProjectODBDeepOnet/train_data --which both --save-plots
"""
import os
import argparse
import numpy as np
import math
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from tqdm import tqdm
    TQDM = tqdm
except Exception:
    TQDM = lambda x: x

# Use canonical VTK writer
from .vtk_io import write_point_vtk
from .output import prepare_output_structure

# -------------------------
# 指标计算
# -------------------------
def compute_metrics(pred, true):
    diff = pred - true
    abs_diff = np.abs(diff)
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff, axis=0)
    max_abs = np.max(abs_diff, axis=0)
    eps = 1e-12
    denom = np.std(true, axis=0)
    rel_rmse = rmse / (denom + eps)
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_abs': max_abs,
        'rel_rmse': rel_rmse
    }

# -------------------------
# 辅助：寻找预测文件
# -------------------------
def find_prediction_paths(data_dir: Path, results_dir: Path):
    res = {}
    res['data_default'] = data_dir / "deeponet_predictions.npy"
    res['baseline'] = results_dir / "baseline_full_predictions.npy"
    res['stacked'] = results_dir / "stacked_full_predictions.npy"
    return res

# -------------------------
# 单个预���文件分析
# -------------------------
def analyze_one(preds_path: Path, tag: str, data_dir: Path, out_dir: Path, samples_arg: str,
                channel_arg: str, save_plots: bool, downsample: int, topk: int):
    print(f"Analyzing predictions: {preds_path}  tag={tag}")
    preds_path = Path(preds_path)
    outputs_path = data_dir / "deeponet_outputs.npy"
    coords_path = data_dir / "deeponet_trunk_coords.npy"
    channels_txt = data_dir / "deeponet_outputs_channels.txt"
    log_csv = data_dir / "training_log.csv"

    for p in [preds_path, outputs_path, coords_path]:
        if not p.exists():
            raise SystemExit(f"缺少文件: {p}")

    preds = np.load(preds_path, mmap_mode='r')
    outputs = np.load(outputs_path, mmap_mode='r')
    coords = np.load(coords_path)
    n_samples, n_nodes, n_channels = outputs.shape
    if preds.shape != outputs.shape:
        raise SystemExit(f"predictions shape {preds.shape} != outputs shape {outputs.shape}")

    if channels_txt.exists():
        with open(channels_txt, 'r') as f:
            channels = [l.strip() for l in f if l.strip()]
    else:
        channels = [f"c{i}" for i in range(n_channels)]

    # parse sample indices
    if samples_arg.strip().lower() == 'all':
        sample_indices = list(range(n_samples))
    else:
        toks = [t.strip() for t in samples_arg.split(',') if t.strip()!='']
        sample_indices = []
        for t in toks:
            try:
                i = int(t)
                if 0 <= i < n_samples:
                    sample_indices.append(i)
            except:
                pass
        if not sample_indices:
            sample_indices = [0]

    # channel selection
    if channel_arg.strip().lower() == 'all':
        channel_indices = list(range(n_channels))
    else:
        tokens = [t.strip() for t in channel_arg.split(',') if t.strip()!='']
        channel_indices = []
        for t in tokens:
            if t in channels:
                channel_indices.append(channels.index(t))
        if not channel_indices:
            channel_indices = list(range(n_channels))

    suffix = f"_{tag}" if tag else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_prefix = out_dir / f"analysis{suffix}"

    # training log plot
    if log_csv.exists():
        try:
            epochs = []; train_loss = []; val_loss = []
            with open(log_csv, 'r') as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    epochs.append(int(row[0])); train_loss.append(float(row[1])); val_loss.append(float(row[2]))
            epochs = np.array(epochs); train_loss = np.array(train_loss); val_loss = np.array(val_loss)
            if HAS_MPL and save_plots:
                plt.figure(figsize=(6,4))
                plt.plot(epochs, train_loss, label='train')
                plt.plot(epochs, val_loss, label='val')
                plt.yscale('log')
                plt.xlabel('epoch'); plt.ylabel('loss (log)')
                plt.legend(); plt.grid(True)
                plt.tight_layout()
                p = analysis_prefix.with_name(analysis_prefix.name + "_training_loss.png")
                plt.savefig(p, dpi=150); plt.close()
                print("Saved", p)
        except Exception as e:
            print("绘制 training_log 失败:", e)
    else:
        print("training_log.csv 不存在，跳过训练曲线绘制。")

    # per-sample & per-channel stats
    summary_by_sample = []
    sum_mse = np.zeros((n_channels,), dtype=np.float64)
    sum_mae = np.zeros((n_channels,), dtype=np.float64)
    sum_max = np.zeros((n_channels,), dtype=np.float64)

    for si in TQDM(sample_indices, desc="Computing sample metrics"):
        pred = np.array(preds[si, :, :], dtype=np.float32)[:, channel_indices]
        true = np.array(outputs[si, :, :], dtype=np.float32)[:, channel_indices]
        metrics = compute_metrics(pred, true)
        overall_rmse = float(np.sqrt(np.mean(metrics['mse'])))
        overall_mae = float(np.mean(metrics['mae']))
        overall_max = float(np.max(metrics['max_abs']))
        summary_by_sample.append({'sample': si, 'rmse': overall_rmse, 'mae': overall_mae, 'max_abs': overall_max})
        sum_mse[channel_indices] += metrics['mse']
        sum_mae[channel_indices] += metrics['mae']
        sum_max[channel_indices] = np.maximum(sum_max[channel_indices], metrics['max_abs'])

    n_considered = len(sample_indices)
    avg_mse = (sum_mse / max(1, n_considered)).astype(np.float64)
    avg_mae = (sum_mae / max(1, n_considered)).astype(np.float64)
    max_abs_overall = sum_max.astype(np.float64)

    # save summary_by_sample
    samp_csv = out_dir / f"summary_by_sample{suffix}.csv"
    with open(samp_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sample','rmse','mae','max_abs'])
        for r in summary_by_sample:
            w.writerow([r['sample'], r['rmse'], r['mae'], r['max_abs']])
    print("Saved", samp_csv)

    # save summary_by_channel
    chan_csv = out_dir / f"summary_by_channel{suffix}.csv"
    with open(chan_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel','avg_mse','avg_rmse','avg_mae','max_abs_over_samples'])
        for idx in channel_indices:
            ch = channels[idx]
            amse = avg_mse[idx]
            armse = math.sqrt(amse)
            amae = avg_mae[idx]
            amax = max_abs_overall[idx]
            w.writerow([ch, amse, armse, amae, amax])
    print("Saved", chan_csv)

    # parity pooled plot
    if HAS_MPL and save_plots:
        try:
            all_true = []; all_pred = []
            for si in sample_indices:
                t = np.array(outputs[si, :, :], dtype=np.float32)[:, channel_indices].reshape(-1)
                p = np.array(preds[si, :, :], dtype=np.float32)[:, channel_indices].reshape(-1)
                all_true.append(t); all_pred.append(p)
            all_true = np.concatenate(all_true); all_pred = np.concatenate(all_pred)
            N = all_true.size
            ds = int(max(1, math.ceil(N / max(1, downsample))))
            idxs = np.arange(0, N, ds)
            t_s = all_true[idxs]; p_s = all_pred[idxs]
            plt.figure(figsize=(5,5))
            plt.scatter(t_s, p_s, s=1, alpha=0.3)
            mn = min(t_s.min(), p_s.min()); mx = max(t_s.max(), p_s.max())
            plt.plot([mn,mx],[mn,mx], 'r--', linewidth=1)
            plt.xlabel('True'); plt.ylabel('Pred')
            plt.title(f'Parity plot (pooled channels) {tag}')
            plt.grid(True); plt.tight_layout()
            p = analysis_prefix.with_name(analysis_prefix.name + f"_parity{suffix}.png")
            plt.savefig(p, dpi=150); plt.close()
            print("Saved", p)
        except Exception as e:
            print("Parity plot failed:", e)

    # compute per-node average absolute error and worst-k nodes
    abs_acc = np.zeros((n_nodes,), dtype=np.float64)
    for si in TQDM(sample_indices, desc="Node error accumulate"):
        pred = np.array(preds[si, :, :], dtype=np.float32)[:, channel_indices]
        true = np.array(outputs[si, :, :], dtype=np.float32)[:, channel_indices]
        abs_node = np.mean(np.abs(pred - true), axis=1)
        abs_acc += abs_node
    abs_acc /= max(1, n_considered)

    worst_idx = np.argsort(-abs_acc)[:topk]
    worst_csv = out_dir / f"worst_nodes{suffix}.csv"
    with open(worst_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['rank','node_index','x','y','z','avg_abs_error'])
        for r, idx in enumerate(worst_idx, start=1):
            x,y,z = coords[idx]
            w.writerow([r, int(idx), float(x), float(y), float(z), float(abs_acc[idx])])
    print("Saved worst nodes list to", worst_csv)

    # node error hist/box
    if HAS_MPL and save_plots:
        try:
            plt.figure(figsize=(6,4))
            plt.subplot(1,2,1)
            plt.hist(abs_acc, bins=100)
            plt.title('node avg abs error histogram')
            plt.xlabel('avg abs error')
            plt.subplot(1,2,2)
            plt.boxplot(abs_acc[np.isfinite(abs_acc)], vert=False)
            plt.title('node avg abs error box')
            plt.tight_layout()
            p = analysis_prefix.with_name(analysis_prefix.name + f"_node_error_hist_box{suffix}.png")
            plt.savefig(p, dpi=150); plt.close()
            print("Saved", p)
        except Exception as e:
            print("绘制节点误差图失败:", e)

    # export VTK for requested samples
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)
    for si in sample_indices:
        pred = np.array(preds[si, :, :], dtype=np.float32)
        true = np.array(outputs[si, :, :], dtype=np.float32)
        diff = pred - true
        point_data = {}
        for ci, name in enumerate(channels):
            point_data[f'pred_{name}'] = pred[:, ci]
            point_data[f'true_{name}'] = true[:, ci]
            point_data[f'diff_{name}'] = diff[:, ci]
        if 'Ux' in channels and 'Uy' in channels and 'Uz' in channels:
            ux = channels.index('Ux'); uy = channels.index('Uy'); uz = channels.index('Uz')
            point_data['pred_U'] = pred[:, [ux,uy,uz]]
            point_data['true_U'] = true[:, [ux,uy,uz]]
            point_data['diff_U'] = diff[:, [ux,uy,uz]]
        vtkfile = vtk_dir / f"sample_{si:04d}{suffix}.vtk"
        write_point_vtk(str(vtkfile), coords, point_data)
    print("Wrote VTKs to:", vtk_dir)

    # return key summary paths
    return {
        "summary_sample_csv": samp_csv,
        "summary_channel_csv": chan_csv,
        "worst_nodes_csv": worst_csv,
        "rmse_per_channel": avg_mse  # return raw for optional further use
    }

# -------------------------
# 主函数（含两组比较）
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="输入数据目录（默认优先使用脚本上级目录的 train_data）")
    parser.add_argument("--results-dir", type=str, default=None, help="预测结果所在目录（默认 project_root/train_results）")
    parser.add_argument("--out-dir", type=str, default=None, help="analysis 输出目录（默认 data_dir/analysis）")
    parser.add_argument("--output-dir", type=str, default=None, help="统一 output 根目录（优先级低于 --out-dir，但高于默认 project_root/output）")
    parser.add_argument("--which", type=str, default="auto", choices=["auto","baseline","stacked","both"],
                        help="选择要分析哪个 predictions: auto/baseline/stacked/both")
    parser.add_argument("--samples", type=str, default="0", help="要分析的样本索引，逗号分隔或 'all'（默认 0）")
    parser.add_argument("--channels", type=str, default="all", help="逗号分隔的通道名或 'all'")
    parser.add_argument("--topk", type=int, default=20, help="列出 top-K 最差节点")
    parser.add_argument("--downsample", type=int, default=100000, help="绘图最大采样点数")
    parser.add_argument("--save-plots", action="store_true", help="保存 PNG 图像（需要 matplotlib）")
    args = parser.parse_args()

    # auto-detect data_dir
    if args.data_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        candidate = project_root / "train_data"
        if candidate.exists():
            data_dir = candidate
        else:
            data_dir = Path.cwd() / "train_data"
    else:
        data_dir = Path(args.data_dir)

    # results dir
    if args.results_dir is None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        results_dir = project_root / "train_data"
    else:
        results_dir = Path(args.results_dir)

    # out dir priority:
    # 1) --out-dir (explicit analysis dir)
    # 2) --output-dir (root) -> output/analysis
    # 3) default: data_dir / "analysis"
    if args.out_dir is not None:
        out_dir_base = Path(args.out_dir)
        out_dir_base.mkdir(parents=True, exist_ok=True)
    else:
        if args.output_dir is not None:
            out_dirs = prepare_output_structure(args.output_dir)
        else:
            out_dirs = prepare_output_structure(None)
        out_dir_base = out_dirs["analysis"]

    print("Using data_dir:", data_dir)
    print("Using results_dir:", results_dir)
    print("Analysis outputs go to:", out_dir_base)

    candidates = find_prediction_paths(data_dir, results_dir)

    selected = []
    if args.which == "auto":
        if candidates['data_default'].exists():
            selected = [("default", candidates['data_default'])]
        else:
            if candidates['baseline'].exists() and not candidates['stacked'].exists():
                selected = [("baseline", candidates['baseline'])]
            elif candidates['stacked'].exists() and not candidates['baseline'].exists():
                selected = [("stacked", candidates['stacked'])]
            elif candidates['baseline'].exists() and candidates['stacked'].exists():
                print("Both baseline and stacked predictions found; analyzing both. Use --which to change.")
                selected = [("baseline", candidates['baseline']), ("stacked", candidates['stacked'])]
            else:
                raise SystemExit("没有找到预测文件。请先运行训练脚本或把预测文件放到 train_results，或使用 --which 指定现有文件。")
    elif args.which == "baseline":
        if candidates['baseline'].exists():
            selected = [("baseline", candidates['baseline'])]
        else:
            raise SystemExit(f"baseline predictions not found at {candidates['baseline']}")
    elif args.which == "stacked":
        if candidates['stacked'].exists():
            selected = [("stacked", candidates['stacked'])]
        else:
            raise SystemExit(f"stacked predictions not found at {candidates['stacked']}")
    elif args.which == "both":
        found = []
        if candidates['baseline'].exists():
            found.append(("baseline", candidates['baseline']))
        if candidates['stacked'].exists():
            found.append(("stacked", candidates['stacked']))
        if not found:
            raise SystemExit(f"No baseline or stacked predictions found in {results_dir}")
        selected = found

    # analyze each selected prediction
    results = {}
    for tag, p in selected:
        res = analyze_one(p, tag, data_dir, out_dir_base, args.samples, args.channels, args.save_plots, args.downsample, args.topk)
        results[tag] = res

    # if both were analyzed, compute compare CSV + optional bar plot
    tags_present = [t for t in results.keys()]
    if ("baseline" in tags_present) and ("stacked" in tags_present):
        print("Generating comparison between baseline and stacked...")
        # load preds + truths
        base_path = candidates['baseline']
        stacked_path = candidates['stacked']
        preds_base = np.load(base_path)
        preds_stacked = np.load(stacked_path)
        outputs = np.load(data_dir / "deeponet_outputs.npy", mmap_mode='r')
        channels_file = data_dir / "deeponet_outputs_channels.txt"
        if Path(channels_file).exists():
            with open(channels_file, 'r') as f:
                channels = [l.strip() for l in f if l.strip()]
        else:
            channels = [f"c{i}" for i in range(outputs.shape[2])]

        # compute per-channel metrics
        dif_base = preds_base - outputs
        dif_stacked = preds_stacked - outputs
        mse_base = np.mean(dif_base**2, axis=(0,1)); rmse_base = np.sqrt(mse_base)
        mse_stacked = np.mean(dif_stacked**2, axis=(0,1)); rmse_stacked = np.sqrt(mse_stacked)
        mae_base = np.mean(np.abs(dif_base), axis=(0,1)); mae_stacked = np.mean(np.abs(dif_stacked), axis=(0,1))
        max_base = np.max(np.abs(dif_base), axis=(0,1)); max_stacked = np.max(np.abs(dif_stacked), axis=(0,1))

        # write CSV
        cmp_csv = out_dir_base / "compare_models_channel_metrics.csv"
        with open(cmp_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["channel","baseline_rmse","stacked_rmse","baseline_mae","stacked_mae","baseline_maxabs","stacked_maxabs"])
            for i,ch in enumerate(channels):
                w.writerow([ch, float(rmse_base[i]), float(rmse_stacked[i]), float(mae_base[i]), float(mae_stacked[i]), float(max_base[i]), float(max_stacked[i])])
        # overall
        overall_base = float(np.sqrt(np.mean(mse_base)))
        overall_stacked = float(np.sqrt(np.mean(mse_stacked)))
        summary_csv = out_dir_base / "compare_models_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(["method","overall_rmse"]); w.writerow(["baseline", overall_base]); w.writerow(["stacked", overall_stacked])

        print("Saved comparison CSVs:", cmp_csv, summary_csv)

        # optional bar plot of RMSE per channel
        if HAS_MPL:
            try:
                ind = np.arange(len(channels))
                width = 0.35
                plt.figure(figsize=(max(6, len(channels)*0.3), 4))
                plt.bar(ind - width/2, rmse_base, width, label='baseline')
                plt.bar(ind + width/2, rmse_stacked, width, label='stacked')
                plt.ylabel('RMSE'); plt.title('Per-channel RMSE: baseline vs stacked')
                plt.xticks(ind, channels, rotation=90, fontsize=max(6, 8 - int(len(channels)/10)))
                plt.legend()
                plt.tight_layout()
                p = out_dir_base / "compare_rmse_per_channel.png"
                plt.savefig(p, dpi=150); plt.close()
                print("Saved plot:", p)
            except Exception as e:
                print("绘制对比条形图失败:", e)

    print("All analysis finished. Outputs in:", out_dir_base)

if __name__ == "__main__":
    main()