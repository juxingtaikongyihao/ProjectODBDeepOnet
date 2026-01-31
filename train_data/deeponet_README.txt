DeepONet data prepared by prepare_data_from_odbs.py

生成文件（位于本目录）：
  deeponet_trunk_coords.npy        (n_nodes x 3)  节点坐标
  deeponet_node_list.pkl           ([(instanceName,nodeLabel), ...])
  deeponet_node_list_preview.txt   (前 200 条节点预览)
  deeponet_outputs.npy             (n_samples x n_nodes x n_channels)  dtype=float32
  deeponet_outputs_channels.txt    (通道顺序)
  deeponet_F_global.npy            (n_samples x ndof)  若脚本成功从 ODB 中提取 RF/CF

说明：
 - 如果需要使用 branch 输入，请准备 loads.npy (n_samples x m)，然后替换占位文件。
 - 若要使用 SE-S 物理损失并计算 Schur 矩阵，需要 K 的列（对 master DOF 逐列扰动获得）。我可提供相应脚本。
 - outputs_channels.txt 列出了 deeponet_outputs.npy 的通道顺序，用于训练时对应输出分量。
