# Full-History attention：完整历史的精确分块计算

`full_history` 默认使用 `segmented` 后端；训练可用 `--history-attention-backend dense` 选择原稠密正确性参照。legacy 路径不变，权重键、位置编码、H/F/A 可见性和 checkpoint schema 不变。加载已有 full-history checkpoint 默认采用分块实现；`load_full_history_checkpoint(..., attention_backend="dense")` 可用于短序列对照。

## 计算方式与边界

历史查询按相同 `(block_id, end_step)` 分组，每组通过一次 SDPA 读取所有满足 block ID、时间截止和有效性条件的历史键。同一历史块内部仍然双向可见，H 不能读取 F，F 读取合法 H 和自身 F，A 读取合法 H、当前 F 和整段 A。只切分查询，不分别对键分块做 softmax；保留完整历史和跨历史梯度，不使用训练期持久 KV，也不做摘要、滑窗或截断。

`memory_attention.py` 在进入层循环前根据布局生成区间计划，每次只构造一个键有效性向量，避免 `N_query × N_key` mask。正常完整历史使用前缀张量视图；有 padding 或不连续有效键时只拼接合法区间。F/A 共用相同键可见性，使用一个无 mask 的 SDPA 调用（每个 batch 行一次）。无合法键或无效查询的输出、梯度均清零。计划不包含可训练张量，activation checkpoint 重算时可重复使用。

Action Expert 逐层按需拼接 H/F，避免提前构造全部层的 H/F 副本，与 attention 为反向传播保存的 H/F/A 同时常驻。拼接保持 autograd，没有 detach，也不改变键顺序；稠密参照保留原路径便于对比。

CUDA BF16 的无 mask SDPA 可使用 FlashAttention；实际内核必须通过 profiler 验证，不能仅根据函数名推断。此实现保留 CPU FP32 路径供参照测试，没有硬性依赖外部 flash-attn 包。非零 dropout 保持相同分布，但分块会改变随机数消耗，不能要求与稠密版本逐位一致。

该优化消除完整 token-pair mask 及注意力分数矩阵的存储需求（融合内核下），不会消除历史 KV、层激活、优化器状态或全历史注意力的平方计算量。查询分块增加 kernel 调用次数；短序列不保证更快。`dense` 仍保留 16,777,216 mask 元素保护；`segmented` 不使用这项稠密限制，实际 GPU 容量仍需测试。`--max-history-blocks` 始终是拒绝超限输入的检查，绝不截断。

## 验证与复现

CPU 回归覆盖稠密/分块的 loss、每层 KV、动作与参数梯度，历史因果、跨样本隔离、不连续 padding、全无效行，以及现有 M4 增量缓存和标准 Trainer 保存/恢复契约。CUDA 检查使用 5×588 个真实形状 token，对比 FP32 数学参照和 BF16 Flash 输出/梯度，并检查完整小模型的 BF16 loss、KV、动作和梯度。误差阈值在脚本中固定；报告实际误差及内核事件。

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="" "$POLICY_PYTHON" -m unittest discover -s tests
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$POLICY_PYTHON" -m scripts.check_memory_attention_cuda \
  --report "$ATTENTION_RUN/cuda_correctness.json"
```

容量诊断使用真实冻结 Encoder 特征、正式 24 层/1024 维 Predictor 和 24 层/512 维 Action Expert，batch=1、BF16、24 层 Predictor activation checkpointing。每个长度独立进程执行至少 3 次 AdamW 更新，计入优化器状态，并单独报告首步和后续更新。它不等同于正式训练或任务成功率评测；计时不含模型加载、特征缓存生成及数据读取。

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$POLICY_PYTHON" -m scripts.benchmark_memory_attention \
  --report "$ATTENTION_RUN/segmented_81.json" --blocks 81 --updates 3 \
  --checkpoint "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" --dataset-dirs "$RMBENCH_PREPARED_DATA" \
  --history-episode-ids 14 --history-feature-cache "$ATTENTION_RUN/features/manifest.json" \
  --output-dir "$ATTENTION_RUN/segmented_81" --history-attention-backend segmented \
  --pred-depth 24 --pred-embed-dim 1024 --pred-num-heads 16 \
  --action-hidden-size 512 --action-num-layers 24 --condition-num-heads 8 \
  --activation-checkpointing-blocks 24 --batch-size 1 --num-workers 0 \
  --mixed-precision bf16 --seed 7
```

缓存须由当前编码代码新建；episode 14 的完整缓存含 81 个观测块和 21 个未来监督决策。5/9/17/33/65/81 块对应决策 16/32/64/128/256/320，均保留从 episode 开始的全部历史。

## 实测结果

2026-10-21，在物理 GPU 1（RTX 5090，32607 MiB，PyTorch 2.7.1+cu128）执行；物理 GPU 0 的原训练保持运行。源码基线为 `a48b71b`，结果对应其上的本轮工作区改动。原始日志、命令、特征、checkpoint 和最终报告位于 HDD 实验目录 `full_history_attention_gpu1_20261021_210000`；本机路径见忽略配置 `configs/attention_gpu1.local.env`。下表为最终代码的独立进程测试，每项 3 次更新，速度取第 2、3 次均值；这些短测数值不应视为长跑吞吐保证。

| 后端 | 历史块 | 历史 token | 峰值 allocated GiB | 峰值 reserved GiB | 秒/更新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense 参照 | 5 | 2940 | 12.50 | 12.84 | 0.536 |
| segmented | 5 | 2940 | 12.50 | 12.79 | 0.468 |
| segmented | 9 | 5292 | 12.56 | 13.05 | 0.552 |
| segmented | 17 | 9996 | 12.71 | 14.61 | 0.845 |
| segmented | 33 | 19404 | 15.34 | 16.63 | 1.645 |
| segmented | 65 | 38220 | 20.71 | 22.79 | 5.122 |
| segmented | 81 | 47628 | 23.41 | 26.04 | 7.478 |

所有长度都通过有限 loss/梯度、AdamW 更新、Predictor/Action Expert 参数变化及 Encoder 冻结检查。allocated/reserved 是 PyTorch 统计，不含全部驱动上下文占用。单卡 32GB 在本配置下能训练当前最长 81 块完整历史，无需先压缩；没有测试 batch>1 或更长轨迹，结论不能外推。

标准 Trainer 另以 81 块、batch=1、gradient accumulation=2，重复 episode 14 的 t=320 决策，完成 6 个 microbatch、3 次更新并保存 checkpoint。峰值 allocated 为 25.72 GiB，reserved 为 27.57 GiB；约 15–16 秒/更新，包含保存的训练调用共 65.94 秒。loss 从 2.4960 到 1.8760，所有参数有限，Encoder 无梯度。这是容量与训练链路诊断，不是成功率或收敛结论；checkpoint 沿用只保存权重/进度、不保存 AdamW moments 的约定。

第一版仅分块 attention、仍提前拼接所有层 H/F 时，81 块单 microbatch 更新峰值 27.80 GiB，但标准 Trainer 的两步累积发生 OOM。最终按层拼接移除了额外常驻 H/F 副本，单 microbatch 峰值降到 23.41 GiB，梯度累积测试通过。首轮失败日志保留在实验根目录，最终通过的日志在 `final/` 子目录；日志中的 `gpu0` 指 `CUDA_VISIBLE_DEVICES=1` 后的逻辑卡 0。

CPU 全量回归 111 项通过；源码环境检查、115 个 Python 文件 AST 解析和 `git diff --check` 通过。CUDA 5 块算子相对 FP32 数学参照：FP32 输出相对 L2 误差 `1.36e-7`，Q/K/V 梯度最大相对 L2 `1.12e-6`；BF16 输出相对 L2 `0.00221`，梯度最大 `0.00279`。完整小模型 BF16 参数梯度相对 L2 `0.00576`、每层 KV 最大相对 L2 `0.00367`、动作相对 L2 `0.000631`，均低于脚本预设阈值。profiler 确认执行 `aten::_scaled_dot_product_flash_attention` 及其 backward。M4 追加历史另补齐 eval 模式和连续 block ID 检查，已有回归同时通过。

checkpoint 经独立进程按 metadata 重建并 strict load，固定噪声动作与保存前结果最大误差为 0，重复调用误差为 0，反归一化动作有限。真实冻结特征通过 21 次追加累积至 81 块，逐层 KV 相对完整重算的最大相对 L2 为 `0.00315`，最终动作相对 L2 为 `0.000973`，低于预设 `0.03`；此阶段峰值 allocated 为 13.65 GiB，21 次特征追加合计 1.46 秒（不含 RGB 编码、数据读取或动作生成，不能当作闭环延迟）。本轮不包含 websocket RGB 闭环、simulator rollout 或 50 条轨迹的正式训练。

## 双 RTX 5090 训练计时（2026-10-22）

在两张 RTX 5090 上通过标准 Trainer/DDP 完成 12 次更新：前 2 次更新的全部 microbatch 都使用最长 81 块，随后 10 次按 Full50 的历史长度分布抽样。保持正式模型大小、BF16、24 层 Predictor activation checkpointing；每卡 batch=1、累积 8，总 batch=16。DDP 默认启用 `gradient_as_bucket_view=True`，并在本次进程设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。未启用这两项时首次尝试在第 2 次最长历史更新 OOM；失败日志保留。卡 0 仍有约 1.6 GiB 桌面占用，训练测试没有终止桌面进程。

本次是吞吐与容量诊断：复用真实 episode 14 的冻结特征、状态和动作，按全部 50 条演示的 1013 个决策长度建立代理索引，并非对 50 条不同演示正式训练。正常计时覆盖 160 个样本，历史平均 42.825 块，略长于全量分布的约 39.54 块；前两次最长历史压力更新不参与平均耗时。每个更新取两 rank 耗时的较大值，计入数据读取、前后向、梯度累积、通信和 AdamW，不含首次建缓存及模型加载。

| 指标 | 实测 |
| --- | ---: |
| 正常更新平均 / 中位数 | 29.52 / 28.63 秒 |
| 正常更新范围（10 次） | 23.18–38.07 秒 |
| 81 块压力更新（第 2 次） | 62.01 秒 |
| 全部测试峰值 allocated / reserved | 27.79 / 29.32 GiB |
| 含加载、保存和退出的测试总时长 | 7.64 分钟 |

两 rank 各完成 96 个 microbatch，最终梯度有限，Encoder 冻结；逐个可训练参数抽取前 8 个元素，共 9760 个元素，在两 rank 上完全相同。最终 checkpoint 保存成功，CPU 检查全部已保存 Predictor/Action Expert 张量有限、权重键无 DDP 前缀且 batch/world-size metadata 正确；两 rank 和 launcher 正常退出。训练集成 CPU 回归 10 项通过。本次没有重复单卡实验已经覆盖的 strict reconstruction/动作一致性测试，也没有进行 rollout。

| 预算 | 更新次数 | 按实测均值估算 |
| --- | ---: | ---: |
| 1 epoch | 63 | 31 分钟 |
| 10 epochs | 630 | 5.17 小时 |
| 20 epochs | 1260 | 10.33 小时 |
| 50 epochs | 3150 | 25.83 小时 |
| 原基线的 21,360 次更新 | 21,360 | 175.15 小时，约 7.30 天 |

当前 stride=16 的完整历史数据每 epoch 有 1013 个决策；DDP/完整累积窗口实际使用 1008 个样本、63 次更新。原 raw 基线有 17,562 个短上下文样本，每 epoch 1097 次更新，21,360 步约为 19.5 epochs；在当前采样下同样的更新数约为 339 epochs。20 epochs 只是在数据遍历次数上接近旧实验，监督决策数和模型上下文已经变化，不能宣称等价的优化预算或足够收敛。

上述估算基于短测和热缓存，建议额外预留约 25% 运行时间，而非视为统计置信区间。它不含 Full50 特征缓存构建、首次冷盘读取及 rollout；Full50 冻结特征预计约 18 GiB，本轮只复用了已有 episode 14 缓存。显存余量仍有限，不能外推到更大 batch、更长历史或被其他 GPU 作业占用的环境。原始命令、每步时间/长度/显存、失败记录、checkpoint 和 `summary.json` 位于 HDD 实验目录 `full_history_dual5090_timing_20261022_101400`，通过记录在其 `bucket_views/`；本机路径和进程环境见忽略配置 `configs/dual5090_timing.local.env`。
