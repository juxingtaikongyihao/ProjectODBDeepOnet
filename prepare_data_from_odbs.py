#!/usr/bin/env python
"""
prepare_data_from_odbs.py

从 Abaqus ODB 文件批量生成 DeepONet 训练所需数据。
将此脚本放在包含样本 .odb 文件的目录中，并用 Abaqus Python 运行：
    abaqus python prepare_data_from_odbs.py

改进点：
- 使用 numpy memmap 分块写入大型数组以降低峰值内存消耗。
- 预先构建元素->全局节点索引映射，加速应力/应变从积分点到节点的平均。
- 将所有生成文件统一输出到 train_data 文件夹中。
- 采用 float32 存储以减少文件体积（可调整）。
- 注释均为中文（按你的要求）。

输出（全部写入 ./train_data）:
- deeponet_trunk_coords.npy         (n_nodes x coord_dim)
- deeponet_node_list.pkl            (python list of (instanceName,nodeLabel))
- deeponet_node_list_preview.txt    (前若干节点人类可读预览)
- deeponet_outputs.npy              (memmap 文件: n_samples x n_nodes x n_channels)  (float32)
- deeponet_outputs_channels.txt     (通道名称顺序)
- deeponet_loads_placeholder.npy    (n_samples x 1) 或 deeponet_F_global.npy (若提取到完整力向量)
- deeponet_frame_field_outputs.txt
- deeponet_README.txt

注意：
- 默认把每个 odb 文件作为一个样本；如果你的样本存放在单个 ODB 的多个 frame 中，请告知，我会修改脚本。
- 默认 DOF_PER_NODE = 3，如为 6 请修改脚本顶部变量。
"""
import os
import sys
import pickle
import math
import traceback
from collections import defaultdict

# Abaqus 专用模块（需在 Abaqus Python 环境中运行）
from odbAccess import openOdb
import numpy as np

# ------------------ 可配置参数（请根据需要修改） ------------------
WORK_DIR = os.getcwd()                          # 工作目录：脚本所在目录
OUT_DIR = os.path.join(WORK_DIR, "train_data")  # 所有输出放到 train_data 文件夹
SKIP_PREFIX = "perturb_col_"                    # 跳过以该前缀开头的 odb（扰动中间文件）
DOF_PER_NODE = 3                                # 每节点自由度（实体常为3；壳/梁或复合模型可能为6）
OUT_DTYPE = np.float32                          # 大数组的数据类型（float32 可节省空间）
MAX_PREVIEW = 200                               # node_list 预览写入的条目数
# ------------------------------------------------------------

# 创建输出目录（如不存在）
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print("工作目录:", WORK_DIR)
print("输出目录:", OUT_DIR)
print("每节点 DOF:", DOF_PER_NODE)
print("输出数据类型:", OUT_DTYPE.__name__)

# 查找目录下的 odb 文件（排除扰动中间文件）
odb_files = [f for f in os.listdir(WORK_DIR) if f.lower().endswith(".odb") and not f.startswith(SKIP_PREFIX)]
odb_files.sort()
if len(odb_files) == 0:
    print("未在目录中发现 .odb 文件。请将样本 .odb 放在此目录后重试。")
    sys.exit(1)

print(f"发现 {len(odb_files)} 个 ODB 文件（将每个视为一个样本）。")

# 辅助函数：安全获取 FieldValue 的实例名
def get_instance_name(v):
    try:
        if getattr(v, "instance", None) is not None:
            return v.instance.name
        return getattr(v, "instanceName", None) or "ASSEMBLY"
    except Exception:
        return getattr(v, "instanceName", None) or "ASSEMBLY"

# ----------------- 从第一个 ODB 建立全局节点列表与坐标 -----------------
first_odb_path = os.path.join(WORK_DIR, odb_files[0])
print("打开首个 ODB 以构建全局节点列表：", first_odb_path)
odb0 = openOdb(first_odb_path)
assembly0 = odb0.rootAssembly

# 按 instance 遍历节点，构建 node_list（(instName, nodeLabel)）和 coords（坐标）
node_list = []
coords = []
instance_node_counts = {}
for inst_name, inst in assembly0.instances.items():
    instance_node_counts[inst_name] = len(inst.nodes)
    for node in inst.nodes:
        node_list.append((inst_name, node.label))
        coords.append(node.coordinates)

n_nodes = len(node_list)
coords_arr = np.array(coords, dtype=np.float64)   # 坐标先用 double 精度

print("节点总数:", n_nodes)
print("部分实例节点数统计：")
for k, v in list(instance_node_counts.items())[:10]:
    print("  ", k, ":", v)

# 保存 trunk coords 与 node_list
np.save(os.path.join(OUT_DIR, "deeponet_trunk_coords.npy"), coords_arr.astype(np.float32))
with open(os.path.join(OUT_DIR, "deeponet_node_list.pkl"), "wb") as f:
    pickle.dump(node_list, f)

# 写入 node_list 预览（便于检查）
with open(os.path.join(OUT_DIR, "deeponet_node_list_preview.txt"), "w") as f:
    f.write("Total nodes: %d\n\n" % n_nodes)
    f.write("First %d node entries (instance, nodeLabel, x,y,z):\n" % MAX_PREVIEW)
    for i, (inst, lbl) in enumerate(node_list[:MAX_PREVIEW]):
        x, y, z = coords[i]
        f.write(f"{i}: {inst}, {lbl}, ({x:.6f}, {y:.6f}, {z:.6f})\n")
print("已保存 trunk_coords、node_list 及预览文件。")

# 构建 node_map： (inst, nodeLabel) -> 全局节点索引
node_map = { (inst, lbl): idx for idx, (inst, lbl) in enumerate(node_list) }

# ----------------- 预计算元素->全局节点索引映射（加速积分点到节点平均） -----------------
print("为每个实例预计算元素 -> 全局节点索引映射（加速应力/应变平均）...")
element_to_global = {}   # instance -> { elementLabel: np.array(global_node_indices, dtype=int) }
for inst_name, inst in assembly0.instances.items():
    emap = {}
    for elem in inst.elements:
        conn = elem.connectivity
        gidxs = []
        for lbl in conn:
            key = (inst_name, lbl)
            idx = node_map.get(key)
            if idx is None:
                idx = node_map.get(("ASSEMBLY", lbl))
            if idx is None:
                continue
            gidxs.append(idx)
        if len(gidxs) > 0:
            emap[elem.label] = np.array(gidxs, dtype=np.int32)
    element_to_global[inst_name] = emap
print("元素映射构建完成，覆盖 %d 个实例。" % len(element_to_global))

odb0.close()

# ----------------- 检查首样本的可用场字段，用于决定输出通道 -----------------
def inspect_fields(odb_path):
    try:
        odb = openOdb(odb_path)
    except Exception as e:
        print("打开 ODB 失败（用于检测字段）:", odb_path, e)
        return []
    step_key = list(odb.steps.keys())[-1]
    frame = odb.steps[step_key].frames[-1]
    fields = list(frame.fieldOutputs.keys())
    odb.close()
    return fields

fields0 = inspect_fields(first_odb_path)
print("首样本帧中可用字段:", fields0)

# 决定通道：优先 U (Ux,Uy,Uz)，其次节点平均 S（6分量），再有则 E（6分量）
channels = []
include_U = 'U' in fields0
include_S = 'S' in fields0
include_E = 'E' in fields0

if include_U:
    channels += ['Ux','Uy','Uz'] if DOF_PER_NODE >= 3 else ['Ux','Uy']
if include_S:
    channels += ['S11','S22','S33','S12','S13','S23']
if include_E:
    channels += ['E11','E22','E33','E12','E13','E23']

n_channels = len(channels)
print("最终输出通道列表:", channels)

# ----------------- 创建 outputs 的 memmap（逐样本写入，避免一次性内存占用） -----------------
n_samples = len(odb_files)
outputs_shape = (n_samples, n_nodes, n_channels)
outputs_path = os.path.join(OUT_DIR, "deeponet_outputs.npy")
outputs_mm = np.lib.format.open_memmap(outputs_path, mode='w+', dtype=OUT_DTYPE, shape=outputs_shape)
print("创建 outputs memmap，形状:", outputs_shape, "dtype:", OUT_DTYPE.__name__)

# 试图收集每个样本的全局力向量（ndof），若无法全部获取则写入 loads 占位
ndof = n_nodes * DOF_PER_NODE
F_global_mm = None
have_F_all = True
loads_placeholder = np.zeros((n_samples, 1), dtype=OUT_DTYPE)

# 保存 sample0 的字段到文件，便于核对
with open(os.path.join(OUT_DIR, "deeponet_frame_field_outputs.txt"), "w") as f:
    f.write("Fields in sample 0 frame:\n")
    for fld in fields0:
        f.write(fld + "\n")

# ----------------- 主循环：逐个打开 ODB 并填充 memmap -----------------
print("开始逐样本处理 ODB 并填充 outputs memmap。样本较多时此过程可能耗时。")
for si, odb_fname in enumerate(odb_files):
    print(f"[{si+1}/{n_samples}] 处理 {odb_fname}")
    odb_path = os.path.join(WORK_DIR, odb_fname)
    try:
        odb = openOdb(odb_path)
    except Exception as e:
        print("  打开 ODB 失败:", odb_fname, e)
        traceback.print_exc()
        continue

    step_key = list(odb.steps.keys())[-1]
    frame = odb.steps[step_key].frames[-1]
    frame_fields = list(frame.fieldOutputs.keys())

    # 1) 提取位移 U（按节点映射）
    U_nodes = None
    if 'U' in frame_fields:
        u_field = frame.fieldOutputs['U']
        u_map = {}
        for v in u_field.values:
            inst_name = get_instance_name(v)
            node_label = getattr(v, "nodeLabel", None)
            if node_label is None:
                continue
            data = v.data
            vals = list(data)[:DOF_PER_NODE] + [0.0]*max(0, DOF_PER_NODE - len(list(data)))
            u_map[(inst_name, node_label)] = vals
        U_nodes = np.zeros((n_nodes, DOF_PER_NODE), dtype=np.float32)
        for idx, (inst, lbl) in enumerate(node_list):
            v = u_map.get((inst, lbl))
            if v is None:
                v = u_map.get(("ASSEMBLY", lbl))
            if v is not None:
                U_nodes[idx,:len(v)] = np.array(v[:DOF_PER_NODE], dtype=np.float32)

    # 2) 从元素 GP 值平均到节点的应力 S（若存在）
    node_stress = None
    if 'S' in frame_fields:
        stress_acc = np.zeros((n_nodes, 6), dtype=np.float64)
        stress_cnt = np.zeros((n_nodes,), dtype=np.int32)
        s_field = frame.fieldOutputs['S']
        for v in s_field.values:
            inst_name = get_instance_name(v)
            elem_label = getattr(v, "elementLabel", None)
            if elem_label is None:
                continue
            elem_map = element_to_global.get(inst_name)
            if elem_map is None:
                elem_map = element_to_global.get('ASSEMBLY', {})
            gidxs = elem_map.get(elem_label)
            if gidxs is None:
                continue
            data = v.data
            comps = np.zeros((6,), dtype=np.float64)
            for k in range(min(6, len(data))):
                comps[k] = data[k]
            stress_acc[gidxs, :] += comps
            stress_cnt[gidxs] += 1
        node_stress = np.zeros((n_nodes, 6), dtype=np.float32)
        nz = stress_cnt > 0
        node_stress[nz, :] = (stress_acc[nz, :] / stress_cnt[nz][:,None]).astype(np.float32)

    # 3) 从元素 GP 值平均到节点的应变 E（若存在）
    node_strain = None
    if 'E' in frame_fields:
        strain_acc = np.zeros((n_nodes, 6), dtype=np.float64)
        strain_cnt = np.zeros((n_nodes,), dtype=np.int32)
        e_field = frame.fieldOutputs['E']
        for v in e_field.values:
            inst_name = get_instance_name(v)
            elem_label = getattr(v, "elementLabel", None)
            if elem_label is None:
                continue
            elem_map = element_to_global.get(inst_name)
            if elem_map is None:
                elem_map = element_to_global.get('ASSEMBLY', {})
            gidxs = elem_map.get(elem_label)
            if gidxs is None:
                continue
            data = v.data
            comps = np.zeros((6,), dtype=np.float64)
            for k in range(min(6, len(data))):
                comps[k] = data[k]
            strain_acc[gidxs, :] += comps
            strain_cnt[gidxs] += 1
        node_strain = np.zeros((n_nodes, 6), dtype=np.float32)
        nz = strain_cnt > 0
        node_strain[nz, :] = (strain_acc[nz, :] / strain_cnt[nz][:,None]).astype(np.float32)

    # 4) 尝试提取全局力向量（RF 或 CF）
    F_global_vec = None
    if 'RF' in frame_fields:
        rf_field = frame.fieldOutputs['RF']
        F_global_vec = np.zeros((ndof,), dtype=np.float32)
        for v in rf_field.values:
            inst_name = get_instance_name(v)
            node_label = getattr(v, "nodeLabel", None)
            if node_label is None:
                continue
            key = (inst_name, node_label)
            gidx = node_map.get(key)
            if gidx is None:
                gidx = node_map.get(("ASSEMBLY", node_label))
            if gidx is None:
                continue
            data = v.data
            for local in range(DOF_PER_NODE):
                if local < len(data):
                    F_global_vec[gidx * DOF_PER_NODE + local] = data[local]
    elif 'CF' in frame_fields:
        cf_field = frame.fieldOutputs['CF']
        F_global_vec = np.zeros((ndof,), dtype=np.float32)
        for v in cf_field.values:
            inst_name = get_instance_name(v)
            node_label = getattr(v, "nodeLabel", None)
            if node_label is None:
                continue
            key = (inst_name, node_label)
            gidx = node_map.get(key)
            if gidx is None:
                gidx = node_map.get(("ASSEMBLY", node_label))
            if gidx is None:
                continue
            data = v.data
            for local in range(DOF_PER_NODE):
                if local < len(data):
                    F_global_vec[gidx * DOF_PER_NODE + local] = data[local]
    else:
        F_global_vec = None

    # 5) 将样本数据按通道顺序写入 outputs memmap
    ch = 0
    if include_U:
        if U_nodes is not None:
            for k in range(min(3, DOF_PER_NODE)):
                outputs_mm[si, :, ch] = U_nodes[:, k].astype(OUT_DTYPE)
                ch += 1
        else:
            ch += min(3, DOF_PER_NODE)
    if include_S:
        if node_stress is not None:
            outputs_mm[si, :, ch:ch+6] = node_stress.astype(OUT_DTYPE)
        ch += 6
    if include_E:
        if node_strain is not None:
            outputs_mm[si, :, ch:ch+6] = node_strain.astype(OUT_DTYPE)
        ch += 6

    # 6) 保存 F_global，如果存在则创建 F_global memmap 并写入；否则写入 loads 占位
    if F_global_vec is not None:
        if si == 0:
            # 首次发现 F，创建 memmap
            F_global_mm = np.lib.format.open_memmap(os.path.join(OUT_DIR, "deeponet_F_global.npy"),
                                                   mode='w+', dtype=OUT_DTYPE, shape=(n_samples, ndof))
        F_global_mm[si, :] = F_global_vec
    else:
        loads_placeholder[si, 0] = 0.0

    odb.close()
    print(f"  已填充样本 {si}: U={U_nodes is not None}, S={node_stress is not None}, E={node_strain is not None}, F={F_global_vec is not None}")

# 释放 memmap，确保数据写回磁盘
del outputs_mm
if 'F_global_mm' in locals() and F_global_mm is not None:
    del F_global_mm
else:
    # 保存 loads 占位文件
    np.save(os.path.join(OUT_DIR, "deeponet_loads_placeholder.npy"), loads_placeholder.astype(OUT_DTYPE))

# 保存通道顺序文件
with open(os.path.join(OUT_DIR, "deeponet_outputs_channels.txt"), "w") as f:
    for c in channels:
        f.write(c + "\n")

# 生成 README 总结
with open(os.path.join(OUT_DIR, "deeponet_README.txt"), "w") as f:
    f.write("DeepONet data prepared by prepare_data_from_odbs.py\n\n")
    f.write("生成文件（位于本目录）：\n")
    f.write("  deeponet_trunk_coords.npy        (n_nodes x 3)  节点坐标\n")
    f.write("  deeponet_node_list.pkl           ([(instanceName,nodeLabel), ...])\n")
    f.write("  deeponet_node_list_preview.txt   (前 %d 条节点预览)\n" % MAX_PREVIEW)
    f.write("  deeponet_outputs.npy             (n_samples x n_nodes x n_channels)  dtype=%s\n" % OUT_DTYPE.__name__)
    f.write("  deeponet_outputs_channels.txt    (通道顺序)\n")
    if os.path.exists(os.path.join(OUT_DIR, "deeponet_F_global.npy")):
        f.write("  deeponet_F_global.npy            (n_samples x ndof)  若脚本成功从 ODB 中提取 RF/CF\n")
    else:
        f.write("  deeponet_loads_placeholder.npy   (n_samples x 1)  占位，请用实际 loads.npy 替换\n")
    f.write("\n说明：\n")
    f.write(" - 如果需要使用 branch 输入，请准备 loads.npy (n_samples x m)，然后替换占位文件。\n")
    f.write(" - 若要使用 SE-S 物理损失并计算 Schur 矩阵，需要 K 的列（对 master DOF 逐列扰动获得）。\n")
    f.write(" - outputs_channels.txt 列出了 deeponet_outputs.npy 的通道顺序，用于训练时对应输出分量。\n")

print("处理完成。所有数据已生成到：", OUT_DIR)