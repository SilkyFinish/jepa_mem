# Full-History v0 M3：训练接入

M3 将 M2 的重算模型接入现有训练循环。默认仍为 `--policy-arch legacy`；只有显式选择 `full_history` 才使用新模型、历史数据集和 checkpoint schema。M0 固定输入契约保持不变，当前版本只支持 full-history；current/window 消融和在线 session 缓存分别留给后续实验及 M4。

## 数据和训练路径

`RMBenchHistoryDataset` 按 `(source_episode, decision_step)` 建立索引，决策点间隔 16，要求 `t+32 < episode_frames`。每个样本包含从 0 到 t、间隔 4 的全部历史块；观测块使用 `[max(0,i-4),i]` 两帧和状态 `q_i`，未来监督使用固定十帧。解码复用官方 RGB codec，三相机顺序、224 图像预处理、whole-task T5 文本和 QUANTILE 统计沿用 M0。原生 `action[t:t+32]` 不再平移一次。

`HistoryBatchCollator` 返回 `rmbench_history_v0` 字典，只在历史块维度右侧补零，并提供 `history_valid`；不跨 episode 拼接 token。训练沿用 Accelerate 的递归设备搬运、AdamW、梯度累积和原 checkpoint 保存循环。按随机池内的历史长度排序、再打乱整批，减少 padding，同时保持每个 epoch 的样本覆盖和 worker 重建时的游标。

`--history-episode-ids` 和 `--history-max-decision-step` 是显式诊断/课程采样参数，会保存进 checkpoint。选择较早决策点时，每个样本的历史前缀仍然完整。`--max-history-blocks` 是拒绝超容量输入的检查，不会截断历史。

## 冻结特征缓存

`scripts/cache_rmbench_history_features.py` 分别保存每个观测块的 `z_i` 和每个监督决策的 `y_future`。观测单独进行两帧编码，不附加 target LayerNorm；未来目标先编码完整十帧，再做额外 LayerNorm，最后选择四组未来 tubelet。缓存中没有 Predictor K/V，可训练 Predictor 在每次更新中重新计算。

manifest 记录固定输入契约、Encoder 权重 SHA256、配置/视频输出分支、编码与预处理源代码哈希（包括缓存脚本本身）、官方 codec 哈希、相机顺序、帧索引、原始 HDF5 哈希、编码与存储 dtype、观测/目标 LayerNorm 规则。训练检查这些信息及文件内容/形状；改变编码代码或权重后应生成新缓存，不能直接沿用旧 manifest。编码使用冻结 Encoder 的 eval + FP32；可选择 FP32 或 BF16 存储，读取时转回 FP32，训练 autocast 单独控制。

## 最小启动示例

先在仓库根目录加载自己的忽略文件，提供 `POLICY_PYTHON`、`VJEPA2_ENCODER_CHECKPOINT`、`TEXT_EMBEDDING_CACHE`、`RMBENCH_PREPARED_DATA` 和 `TRAIN_GPU`。预处理 manifest 和训练统计沿用现有 RMBench 准备流程。下面使用新输出目录，训练一个小模型，验证两条轨迹的早期决策；它不是正式全量训练配置。

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export M3_RUN_DIR="$PWD/runs/memory_v0_m3_example"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$POLICY_PYTHON" scripts/cache_rmbench_history_features.py \
  --cache-output "$M3_RUN_DIR/features" \
  --checkpoint "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
  --dataset-dirs "$RMBENCH_PREPARED_DATA" \
  --history-episode-ids 0 1 --history-max-decision-step 16

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$POLICY_PYTHON" scripts/train_vjepa_policy.py \
  --policy-arch full_history \
  --checkpoint "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
  --dataset-dirs "$RMBENCH_PREPARED_DATA" \
  --history-feature-cache "$M3_RUN_DIR/features/manifest.json" \
  --history-episode-ids 0 1 --history-max-decision-step 16 \
  --output-dir "$M3_RUN_DIR/train" \
  --pred-depth 2 --pred-embed-dim 128 --pred-num-heads 2 \
  --action-num-layers 2 --action-hidden-size 64 --condition-num-heads 2 \
  --activation-checkpointing-blocks 0 \
  --batch-size 1 --num-workers 0 --mixed-precision no \
  --max-steps 4 --save-every 2 --log-every 1 --save-initial-checkpoint
```

去掉 `--history-feature-cache` 即从原始视频实时编码。Encoder 和 T5 始终冻结；From Scratch 指 Predictor 与 Action Expert 随机初始化。默认大模型仍沿用原训练入口的宽度和深度，例子中的 2 层/128 维仅用于诊断。小深度必须相应设置 `--activation-checkpointing-blocks`，不能保留超过模型层数的默认值。

## 保存、恢复和加载

保留原 `predictor / action_expert / policy_serving / model_topology` 字段，新增顶层 `policy_arch=full_history`、`model_version=v0` 和 M0 原样定义的 `memory_schema`。`memory_training` 另存实际模型/RoPE/flow 设置、冻结 Encoder 指纹、数据/文本/统计/特征缓存来源及诊断样本选择。没有写入 session KV。

`--resume PATH` 在 strict load 前检查架构、输入契约、位置/attention 设置、冻结 Encoder、归一化统计与文本缓存。恢复权重、step 和学习率进度，沿用原仓库约定：**不恢复 AdamW moments，也不保证精确恢复随机数和数据游标**。允许显式改变课程样本范围；统计和文本语义必须一致。没有架构标记的旧 checkpoint 仍按 legacy 处理。legacy loader 遇到 full_history 会拒绝，不能仅因权重形状相同就加载。

```python
from vjepa_policy.memory_checkpoint import load_full_history_checkpoint

model = load_full_history_checkpoint(
    checkpoint_path, encoder_checkpoint, device="cuda",
)
normalized_actions = model.predict_action(observation_batch, initial_noise=noise)
# [B,32,14]；使用训练保存的 dataset_stats.json 反归一化为原生关节命令。
```

这个加载器接受完整观测 batch，可用于重算推理验证；在线 RGB 请求协议和持久 KV session 尚未接入。带缓存特征的训练 batch 可直接用于推理，监督字段不参与动作生成。

可选 `--predictor-init` 仍只迁移 Predictor，Action Expert 新建。full_history 额外要求来源 checkpoint 的 packed state、RoPE 配对和插值 metadata 完整且匹配；相机 embedding 数量不同时必须显式指定 `--predictor-view-map`，其余参数严格加载。裸权重无法证明上述语义，会被拒绝。已有双向 Predictor 的参数可作迁移初始化，不代表已经训练好记忆能力。

## 当前容量与验证边界

M1/M2 使用密集布尔 mask 的正确性参考后端，每个 mask 上限为 16,777,216 个元素。M3 在训练前按所选最大完整前缀和 micro-batch 检查容量。224 三视角、batch=1 时，决策 16 的 5 块历史可用；决策 32 的 9 块历史已超过限制。增加梯度累积不会增加单个 micro-batch 的 mask 大小，但不能解决单条长历史容量问题。全 81 块前缀训练还需要分块/长序列 attention 后端，当前不会静默丢弃旧历史。

数值测试运行命令为 `python -m unittest discover -s tests -v`；M3 增加原生帧/动作对齐、右侧 padding、显式采样、缓存重建与拒绝不匹配来源、长度分桶、legacy 入口、checkpoint 错架构拒绝、真实 Trainer update/save/load/resume 等覆盖。实际 GPU 验证结果另记在本文件末尾；小样本拟合不等同于 RMBench 闭环成功率。

## 本轮实测记录（2026-10-21）

工作分支为 \`full-history-v0\`。本轮完整 policy 环境 94 项测试通过；标准库环境 32 项通过、6 个依赖训练环境的模块跳过；Python AST 和 \`git diff --check\` 通过。原版同权重数值回归与默认 legacy 入口均有覆盖。已有 M2 测试发现噪声参数互斥、缓存形状和 production factory 几何检查缺失，本轮补回这些边界检查，正常输入的计算路径保持不变。

完整 Full50 数据索引实际得到 50 条轨迹、1013 个决策、最长 81 块，没有截断；这里只验证索引和原生数值对齐，没有运行全长度训练。GPU 诊断使用实际预训练 ViT-L 和已缓存 T5，选 episode 0/1、决策 0/16，共 4 个样本，使用 2 层/128 维 Predictor 和 64 维 Action Expert，batch=1、严格 FP32。缓存生成、4 步标准 CLI 训练、原始 RGB 从 step 4 续训到 step 5 均实际运行。

小样本拟合先训练 256 步，再通过标准 \`--resume\` 接口训练到 step 2048。固定 flow 噪声与 t=0.5，在全部 4 个训练样本上的平均结果如下；采样动作指标另使用固定初始噪声和 10 步生成。

| 指标 | 随机初始化 | step 2048 |
| --- | ---: | ---: |
| World L1 | 0.663163 | 0.442425 |
| Action flow MSE | 2.354366 | 0.006737 |
| 采样动作 MSE（归一化空间） | 2.287889 | 0.002156 |
| Future cosine similarity | -0.009424 | 0.776925 |

动作已能拟合这 4 个样本；world 分支有下降，但尚未充分拟合。诊断脚本额外要求 world loss 降到初值的 60% 以下，实际比例为 66.7%，因此该附加门槛未通过，原失败日志保留。此项不能宣称“所有 loss 都已过拟合”，也不代表真实任务成功率；正式结构的宽度/深度、长历史训练和泛化仍需后续实验。

保存后按 metadata 重建模型并 strict load，固定输入/噪声的动作最大误差为 0，输出为有限的 \`[1,32,14]\`；Encoder 没有梯度。相同编码 batch 大小时，真实 RGB 重算与缓存的观测及目标特征最大误差均为 0；将多个观测块一起送入 ViT-L 时，由 GPU 数值计算产生的观测特征最大差异约为 \`8.42e-4\`，不应要求不同编码 batch 大小逐位一致。这个小模型缓存训练的 PyTorch 峰值已分配显存约为 1.59 GiB，不能外推为正式模型或 81 块历史的显存需求。

本机证据保存在忽略目录 \`runs/memory_v0_m3/\`：\`policy_tests_final.log\`、\`source_tests.log\`、\`full50_index.json\`、\`features_v2/manifest.json\`、\`gpu_smoke_v2.log\`、\`gpu_raw_resume.log\`、\`fitting_result.json\`、\`gpu_validation_v2.log\`。诊断命令和脚本也保存在同目录，未修改已有实验资产。

原始 RGB 续训后的 checkpoint 另经独立进程重建、strict load 和动作请求验证，归一化及反归一化动作均有限；固定噪声重复请求误差为 0，legacy loader 正确拒绝该 checkpoint。结果为 \`raw_roundtrip.json\`，全阶段汇总为 \`m3_summary.json\`。
