# 共享存储上的八卡 A100 训练

本分支面向“常驻主机保存代码和数据，临时八卡训练节点直接挂载同一路径”的部署。常驻主机完成一次 CPU 准备；平台启动训练节点后，前台执行 `node.sh`。默认联合流程依次完成缓存、实机预检、训练和单卡串行 20 次 rollout；设置 `JEPA_TRAIN_ONLY=1` 或显式传入 `--train-only`，则节点只做 GPU/缓存检查、最长历史压力测试和训练，全部仿真与 rollout 留到常驻主机另行执行。不要在节点入口加 `nohup` 或 `&`，否则平台可能在入口退出后释放节点。所有环境、数据、缓存、checkpoint 和日志放在 `JEPA_SHARED_ROOT`；只有临时 IPC socket 放 `/tmp`。

## 常驻主机只需准备一次

下面路径来自 MemoryL 的历史记录，必须确认当前共享挂载确实存在。新代码和资产使用独立目录，不覆盖 MemoryL、旧数据和已验证的仿真环境。

```bash
git clone --branch full-history-a1008 https://github.com/SilkyFinish/jepa_mem.git /mnt/data/dylan/d2/memory_l/jepa_mem
cd /mnt/data/dylan/d2/memory_l/jepa_mem
bash scripts/a1008/configure.sh \
  /mnt/data/dylan/d2/jepa_mem_assets \
  /mnt/data/dylan/d2/memory/RMBench \
  /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python
# 如果 conda 不在 PATH 中：
export CONDA_EXE=/mnt/data/WORKSPACE/anaconda3/bin/conda
bash scripts/a1008/host_setup.sh
```

`configure.sh` 仅写入 Git 忽略的 `configs/a1008.local.env`。需要改变路径、运行名或配置时编辑该文件；不要修改 Python。`host_setup.sh` 建立独立 Python 3.10 conda 环境，默认 PyTorch 2.7.1 / CUDA 12.6，固定核心依赖并运行 `pip check`，记录完整依赖列表。若节点驱动要求其他 wheel，可在本地配置设 `JEPA_TORCH_INDEX` 后安装兼容的同版本 PyTorch；GPU/CUDA/NCCL 实测仍是必需步骤。

默认联合流程下，安装器另外建立只含 websocket 客户端依赖的 `envs/sim_client` 覆盖目录，不改动原仿真环境；RMBench 原有 SAPIEN、CuRobo、机器人、对象资产、Vulkan/渲染驱动仍须能在训练节点使用。只训练模式跳过此 overlay 安装，节点也不要求设置或挂载 `RMBENCH_ROOT`、`RMBENCH_PYTHON`；仿真环境仅需在后续实际评测的主机可用。不得把其他 GPU 的 CUDA/架构/OIDN 补丁直接套到 A100。

官方资产锁定在 [a1008-assets.json](../configs/a1008-assets.json)：RMBench V1 `data/put_back_block/demo_clean.zip`（836 MB）、官方 RGB codec、V-JEPA 2.1 ViT-L 编码器，以及 T5-v1_1-XXL 固定 revision。下载后校验 SHA256。数据逐帧验证并生成本机路径的 manifest 和 QUANTILE 统计，要求 50 episodes、17,562 frames、1,013 个合法决策、最长 81 历史块。不会读取服务器旧版 `demo_clean/data/episode0.hdf5`。T5 原始 PyTorch 权重约 44.5 GB，下载较慢时可重跑准备命令续传；不要重复下载 TF 权重。

建议共享目录预留至少 150 GB（安装器要求至少 120 GiB 空闲），包括环境、下载、约 18 GiB 全量 FP32 视觉缓存、约 7.2 GB 完整续训状态，以及若干约 2.4 GB 推理权重。原子替换完整状态时还需一份临时空间。中断的缓存和实验保留为独立目录，可检查后自行清理。

## 本地 BERT 语言特征变体

原始 T5 配方保持默认；使用本地 BERT 时，在 Git 忽略的 `configs/a1008.local.env` 中加入以下配置，然后运行同一个 `host_setup.sh`：

```bash
export JEPA_TEXT_ENCODER=bert
export JEPA_BERT_SOURCE=/path/to/existing/bert-base-uncased
export JEPA_RUN_NAME=put_back_block_full50_bert_ep100
```

此变体标记为 `model_version=v0_bert`，冻结 BERT-base-uncased 并缓存完整 prompt 的 128×768 逐 token 特征与 mask；不使用句向量池化，不补零到 4096 维，也不训练随机的离线投影。Predictor 和 Action Expert 现有的可训练语言输入层直接接收 768 维特征，其余视觉编码器、历史、采样、损失和全局 batch 不变。这是 From Scratch 的 BERT 变体，不是原 T5 配方的等价复现，不能混用两者的 checkpoint 续训。

准备脚本只读 `JEPA_BERT_SOURCE`，按资产锁校验并独立复制到 `$JEPA_SHARED_ROOT/weights/bert-base-uncased`；不会下载 T5，也不修改已有 TurboVLA 权重。BERT 的五个本地文件摘要固定在 `a1008-assets.json`，作为该本地快照的来源标识。常驻主机用 CPU 生成指令缓存；训练和 rollout 只读取缓存，不常驻 BERT。BERT 使用独立的 `text_cache_bert/`、`history_features_bert/`、`assets_ready_bert.json` 和运行目录，保留原 T5 资产和实验。官方 full50 数据、codec、V-JEPA 权重和独立环境仍须准备，安装器仍要求至少 120 GiB 空闲空间。

`node.sh ... --plan` 同时输出语言编码器、模型版本和特征宽度；BERT 应显示 `bert`、`v0_bert`、`768`。节点启动命令和八卡预检顺序不变；已在常驻主机生成的 BERT 文本缓存会直接复用，视觉缓存仍在节点 GPU 上构建。语言缓存和 checkpoint 均校验来源，原版 T5 仍使用 `.t5_len128.pt` 文件，BERT 使用 `.bert_len128.pt` 文件。

## 平台中的启动命令

在分配到八卡 A100 的节点填写这一条命令；共享路径与常驻主机保持相同：

```bash
bash /mnt/data/dylan/d2/memory_l/jepa_mem/scripts/a1008/node.sh
```

也可明确传入配置：`bash /path/to/jepa_mem/scripts/a1008/node.sh /path/to/jepa_mem/configs/a1008.local.env`。脚本尊重平台设置的 `CUDA_VISIBLE_DEVICES`，要求可见设备数为 8，不会重设为宿主机卡号。需要先审阅配置，用 `node.sh configs/a1008.local.env --plan`；只完成实机准备和预检，用 `--preflight-only`。准备失败返回非零状态，正式训练不会偷偷继续。

默认联合流程的启动顺序如下：

1. 记录源码 revision、依赖、GPU 型号、显存、拓扑和 NCCL 环境，实际进行八卡 NCCL all-reduce 并正常退出。
2. 在第一个已分配 GPU 上生成 128-token T5 缓存，再从全部 50 条轨迹生成冻结 Encoder 视觉缓存；缓存中不含可训练 Predictor KV。
3. 按实际全量索引核算训练计划，检查 feature-cache 源码/Encoder/数据/codec 指纹。
4. 用正式 24 层模型、最长完整历史和相同全局 batch 做两次 AdamW 更新（第二步包含优化器/通信桶的显存开销）；此项为重复最长样本诊断，不用于正式训练初始化。
5. 在原仿真环境验证渲染、native expert 和示范回放，再实际 strict-load 诊断 checkpoint，完成 1 次 32 步 websocket 闭环。dense15 回放是数据接口检查；正式得分使用 native_qpos。
6. 从随机 Predictor/Action Expert 初始化开始正式训练，或从同一个运行的 `resume_state.pt` 恢复；训练成功后释放八卡训练进程，在第一个已分配 GPU 上串行完成 20 次 native_qpos rollout。

## 八卡节点只训练，每卡 batch 2

如果仿真计划在常驻主机完成，在本地配置中明确设置：

```bash
export JEPA_TRAIN_ONLY=1
export JEPA_PER_DEVICE_BATCH=2
export JEPA_RUN_NAME=put_back_block_full50_bert_b2_trainonly_ep400
export JEPA_EPOCHS=400
```

对应的前台启动命令为：

```bash
bash /mnt/data/dylan/d2/memory_l/jepa_mem/scripts/a1008/node.sh \
  /mnt/data/dylan/d2/memory_l/jepa_mem/configs/a1008.local.env --train-only
```

先审阅计划可追加 `--plan`；只完成节点预检可追加 `--preflight-only`。只训练模式保留 GPU 型号/显存检查、八卡 NCCL、资产来源和缓存校验，以及当前 batch 下最长 81 块历史的两次 AdamW 更新；跳过 SAPIEN 渲染、expert/示范回放、32-step websocket smoke 和训练后自动 20 次 rollout。`status.json` 的预检结果明确记录 `simulator_preflight=skipped`，训练完成时记录 `evaluation=deferred_to_host`；这不代表仿真已经通过，也不会自动在本机启动仿真。节点仍需挂载代码、policy 环境及全部训练资产，但不依赖旧仿真解释器、checkout 或 EGL/Vulkan。

样本分配从 `8 × 1 × 2` 调为 `8 × 2 × 1`，global batch 始终为 16，每 epoch 63 次更新/1,008 个样本。上述本地配置将预算延长为 400 epochs，共 25,200 次更新（公共默认仍为 100 epochs/6,300 次更新）。每 5 epochs（315 updates）保存 checkpoint，同时更新完整续训文件 `resume_state.pt`；权重文件按最近两份的窗口清理，但额外保护第 20、50 和最终第 400 epoch 的里程碑权重，并非永久保留全部每 5 epoch 的文件。更大的单次 batch 可能减少调用开销并提高 GPU 利用率，但可变历史 padding 可能抵消收益，不保证两倍加速，也不保证浮点或随机性逐位相同。batch 2 的显存与吞吐必须在目标节点重新实测；原 batch 1 压力测试不能代替本次验收。最长历史压力测试失败时不会进入正式训练。

batch/累积发生变化必须使用新运行名，不复用旧运行的 `resume_state.pt`，也不把诊断 checkpoint 当作正式初始化。冻结资产和指纹相符的视觉缓存可复用。后续常驻主机评测使用正式 checkpoint、匹配的 BERT 文本缓存和训练数据统计，并单独准备仿真/serving 环境；此模式不新增单卡评测入口。`--eval-only` 与只训练模式不兼容，若使用原联合评测入口，须另备设置 `JEPA_TRAIN_ONLY=0` 的配置并满足其 world size/plan 等校验；外层临时环境变量不能覆盖本地配置中的显式赋值。

## 固定训练预算

| 配置 | 值 |
| --- | --- |
| 变体 | From Scratch；视觉和所选 T5/BERT 编码器冻结 |
| 数据 | put_back_block 官方 full50 |
| 历史 | 完整 episode 前缀，间隔 4，决策间隔 16，不压缩、不截断 |
| 模型 | Predictor 24×1024，Action Expert 24×512 |
| Attention / precision | exact segmented SDPA / BF16 |
| 初始 activation checkpointing | 全部 24 层，优先保证 40/80 GB A100 容量 |
| Batch | 默认 8 GPU × 每卡 1 × 累积 2 = 16；batch 2 配置为 8 × 2 × 1 = 16 |
| Epoch / 更新数 | 100 epoch，63 updates/epoch，共 6,300 updates |
| 每 epoch 实际样本 | 1,008；不足完整累积窗口的尾部 5 个随机样本丢弃 |
| 保存 | 每 5 epoch；另外保留 20、50、100 epoch 权重 |
| 评测 | 联合模式训练后自动 20 次，种子从 100000 开始，native expert-solvable 筛选；只训练模式留待主机独立执行 |

100 epoch 是本实验的初始预算，不是论文宣称的收敛要求。八卡 A100 的耗时仍需实测；历史估计不能作为部署验收。正式训练的 `train/progress.json` 给出本次进程的 step 耗时和剩余 ETA，首个 epoch 后更有参考价值。其他配置通过 `configs/a1008.local.env` 设置；换训练预算/数据/全局配置需新 `JEPA_RUN_NAME`，不能冒充同一训练的续跑。

评测使用仿真器原生步数限制（`--max-steps 0`），不是客户端默认的 32 步诊断。服务器按该限制分配完整历史容量；put_back_block 的 500 步配置对应最多 126 块、约 6.8 GiB 常驻 BF16 KV。预算另外覆盖原子扩容期间同时存活的旧缓存、新缓存和增量块，约 14.3 GiB；这不包含模型和计算临时显存。KV 达到容量会明确报错，不淘汰历史。输入始终使用训练数据中的固定 whole-task instruction，不提供子任务或目标 pad 标签。

## 进度、停止与节点重建

默认运行目录是 `$JEPA_SHARED_ROOT/runs/put_back_block_full50_ep100`。`status.json` 记录阶段/错误；每次启动产生新的 `attempt-*`，保存命令、日志、预检和评测结果。`train/progress.json`、`train/training_result.json` 和 `train/resume_state.pt` 是训练进度和恢复依据。运行锁拒绝两个节点同时写入同一实验。

```bash
# 常驻主机上查看阶段和训练进度
cat /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_ep100/status.json
cat /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_ep100/train/progress.json
# 请求在下一次 optimizer update 后保存并停止
touch /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_ep100/STOP
```

节点收到 SIGTERM/SIGINT 也会请求保存；平台须留出足够的终止宽限时间（建议至少 5 分钟）。突然断电或 SIGKILL 只能恢复最后一次已提交 checkpoint，最多回退约 5 epoch。停止后的训练不会自动进入评测。确定要恢复时删除自己创建的 `STOP`，在新八卡节点执行相同启动命令。正式训练已完成时，联合模式默认只重新评测，只训练模式校验完成步数后直接记录完成、不启动评测；失败/中断的 rollout 保留，重跑使用全新的 20 次试验，不混合分数。

新增 `--save-training-state` 会原子保存 AdamW moments、scheduler、各 rank 的 Python/NumPy/PyTorch/CUDA RNG、epoch 和已消费的数据游标；`--resume train/resume_state.pt` 恢复这些内容。要求 world size、batch、累积、数据 provenance、随机种子和总 schedule 一致。精确续训面向本实验确定性的缓存数据管线，不保证任意用户自定义随机 worker 数据增强的逐位重现。`checkpoint_step*.pt` 继续保持轻量推理格式；用它续训仍是历史的“权重和步数恢复”，不是完整 AdamW 续跑。

## 本机和远端验证边界

本机 CPU 检查可验证源代码、回归测试及缩小模型的 optimizer 保存/中断/恢复；其他 RTX 硬件或历史配置上的验证不能代替本次 A100 验收。A100 wheel/驱动、八卡 NCCL、共享文件系统性能和当前 batch 的最长历史显存必须在目标节点实测；只有联合模式还会在节点验证 RMBench 仿真，只有实际执行过的主机评测才能报告仿真结果。本机通过不代表远端检查已经通过，新增入口的 CPU/mock 测试也不构成 batch 2 的实机容量或速度证明。既有配置的历史实测见 [2026-10-22 验证记录](a1008_validation_20261022.md) 和 HDD 日志，不应当作本次新配置的验收。

资产来源：[RMBench 官方数据](https://huggingface.co/datasets/TianxingChen/RMBench/tree/main/data/put_back_block)、[官方下载器](https://github.com/RoboTwin-Platform/RMBench/blob/main/data/_download.py)、[Meta V-JEPA](https://github.com/facebookresearch/vjepa2)、[T5](https://huggingface.co/google/t5-v1_1-xxl)、[PyTorch 版本安装说明](https://pytorch.org/get-started/previous-versions/)。
