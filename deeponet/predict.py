#!/usr/bin/env python3
"""
predict.py

Prediction helpers extracted from train.py:
- evaluate_full_samples
- predict_all
- predict_all_and_save
"""
import numpy as np
import torch
from tqdm import tqdm

def evaluate_full_samples(model, outputs_memmap, coords, branch_inputs, sample_idx, device,
                          batch_nodes=4096, target_mean=None, target_std=None):
    model.to(device)
    model.eval()
    n_nodes = coords.shape[0]
    coord_t_all = torch.from_numpy(coords.astype(np.float32)).to(device)
    trunk_feats = []
    with torch.no_grad():
        for i in range(0, n_nodes, batch_nodes):
            c = coord_t_all[i:i+batch_nodes]
            trunk_feats.append(model.trunk(c).cpu().numpy())
    trunk_feats = np.vstack(trunk_feats)  # (n_nodes, R) representation
    total_mse = 0.0
    count = 0
    n_channels = outputs_memmap.shape[2]
    R = model.rank
    if target_mean is None or target_std is None:
        tmean = np.zeros((n_channels,), dtype=np.float32)
        tstd = np.ones((n_channels,), dtype=np.float32)
    else:
        tmean = target_mean
        tstd = target_std

    for si in sample_idx:
        branch = branch_inputs[si:si+1]
        branch_t = torch.from_numpy(branch.astype(np.float32)).to(device)
        with torch.no_grad():
            coeffs = model.branch(branch_t).cpu().numpy()
        coeffs = coeffs.reshape(1, n_channels, R)
        out = np.einsum('bcr,rn->bcn', coeffs, trunk_feats.T)
        out = np.transpose(out, (0,2,1)).astype(np.float32)
        out_denorm = out * tstd[None, None, :] + tmean[None, None, :]
        gt = np.array(outputs_memmap[si, :, :], dtype=np.float32)
        mse = np.mean((gt - out_denorm[0]) ** 2)
        total_mse += mse
        count += 1

    if count == 0:
        return float('nan')
    return total_mse / count


def predict_all(model, coords, branch_inputs, out_path, device, batch_samples=8, batch_nodes=4096,
                target_mean=None, target_std=None):
    model.to(device)
    model.eval()
    n_samples = branch_inputs.shape[0]
    n_nodes = coords.shape[0]
    n_channels = model.n_channels
    coord_t_all = torch.from_numpy(coords.astype(np.float32)).to(device)
    trunk_feats = []
    with torch.no_grad():
        for i in range(0, n_nodes, batch_nodes):
            c = coord_t_all[i:i+batch_nodes]
            trunk_feats.append(model.trunk(c).cpu().numpy())
    trunk_feats = np.vstack(trunk_feats)
    if out_path:
        preds_mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32,
                                            shape=(n_samples, n_nodes, n_channels))
    else:
        preds_mm = np.zeros((n_samples, n_nodes, n_channels), dtype=np.float32)
    if target_mean is None or target_std is None:
        tmean = np.zeros((n_channels,), dtype=np.float32)
        tstd = np.ones((n_channels,), dtype=np.float32)
    else:
        tmean = target_mean
        tstd = target_std

    for si0 in tqdm(range(0, n_samples, batch_samples), desc="Predicting samples"):
        si1 = min(n_samples, si0 + batch_samples)
        branch_batch = torch.from_numpy(branch_inputs[si0:si1].astype(np.float32)).to(device)
        with torch.no_grad():
            coeffs = model.branch(branch_batch).cpu().numpy()
        Bs = coeffs.shape[0]
        coeffs = coeffs.reshape(Bs, n_channels, model.rank)
        out_block = np.einsum('bcr,rn->bcn', coeffs, trunk_feats.T)
        out_block = np.transpose(out_block, (0, 2, 1)).astype(np.float32)
        out_block = out_block * tstd[None, None, :] + tmean[None, None, :]
        preds_mm[si0:si1, :, :] = out_block

    return preds_mm

def predict_all_and_save(model, coords, branch_inputs, device, out_path, batch_samples=8, batch_nodes=4096,
                         target_mean=None, target_std=None):
    preds = predict_all(model, coords, branch_inputs, out_path, device, batch_samples=batch_samples, batch_nodes=batch_nodes,
                        target_mean=target_mean, target_std=target_std)
    if out_path:
        # predict_all already wrote to memmap; ensure flush by closing reference
        return out_path
    else:
        # no out_path: predictions in-memory; caller must handle
        return preds