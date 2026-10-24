# 共享存储上的八卡 A100 训练

本分支保留服务器共享路径、独立 conda 环境、本地 BERT 和可选只训练流程，并同步新版计算与采样优化。默认配方是 **From Scratch、BERT、每视角 14×14 历史特征、逐帧决策采样、25 epochs**，模型版本为 `v2_full_bert`。视觉 Encoder 和 BERT 冻结；Predictor 24×1024、Action Expert 24×512，历史保持完整 episode 前缀，没有空间池化或历史淘汰。

## 已配置服务器的升级

在现有 `full-history-a1008` checkout 拉取更新，沿用 Git 忽略的 `configs/a1008.local.env` 与已准备的共享资产：

```bash
git pull --ff-only origin full-history-a1008
bash scripts/a1008/node.sh configs/a1008.local.env --dense-full --plan
```

计划应显示 `v2_full_bert`、`history_spatial_pool=1`、17,562 samples、1,097 updates/epoch、25 epochs、27,425 updates 和 global batch 16。`--dense-full` 在读取本地机器配置后选定这套 BERT/14×14/25-epoch 配方，覆盖旧文件中的 v0、100/400 epochs、batch 和旧运行名；本地文件本身不会改写，机器路径、BERT 来源、缓存偏好和平台通信配置继续沿用。每次启动或恢复都保留这个参数；如需另起相同配方实验，可设置 `JEPA_DENSE_FULL_RUN_NAME`。不传此参数时，本地显式赋值仍优先，可按下面配置自定义训练预算或容量设置：

```bash
export JEPA_TEXT_ENCODER=bert
export JEPA_HISTORY_MODEL_VERSION=v2_full
export JEPA_HISTORY_BACKEND=flex
export JEPA_SHARED_PREFIX=1
export JEPA_ACTION_CHECKPOINTING=1
export JEPA_PER_DEVICE_BATCH=2
export JEPA_EPOCHS=25
export JEPA_RUN_NAME=put_back_block_full50_v2_full_bert_ep25
export JEPA_PACK_FEATURES=1
export JEPA_TRAIN_ONLY=1
```

新版使用独立运行目录。采样、训练计划或配方不匹配时，入口在缓存与 GPU 工作之前拒绝复用旧目录。旧实验和 checkpoint 保留；新采样不自动继承旧 v0 的 AdamW、step 或 schedule。跨配方权重初始化需单独校验，不能视为精确续训。

平台前台入口保持原脚本，不加 `nohup` 或 `&`；平台据此前台进程维持节点生命周期：

```bash
bash /mnt/data/dylan/d2/memory_l/jepa_mem/scripts/a1008/node.sh \
  /mnt/data/dylan/d2/memory_l/jepa_mem/configs/a1008.local.env --dense-full --train-only
```

`--plan` 只解析配置，不要求 GPU 或 torch；`--preflight-only` 完成实际缓存与 GPU 容量检查后退出。联合训练/评测模式用 `JEPA_TRAIN_ONLY=0`，另需可用的 RMBench 仿真挂载与解释器；`--eval-only` 与只训练模式互斥。

## 新服务器准备

常驻主机保存代码、数据、环境和权重，临时训练节点挂载同一路径。已有配置和资产无需重复克隆或下载；第一次部署可使用：

```bash
git clone --branch full-history-a1008 https://github.com/SilkyFinish/jepa_mem.git /mnt/data/dylan/d2/memory_l/jepa_mem
cd /mnt/data/dylan/d2/memory_l/jepa_mem
bash scripts/a1008/configure.sh \
  /mnt/data/dylan/d2/jepa_mem_assets \
  /mnt/data/dylan/d2/memory/RMBench \
  /mnt/data/dylan/d2/memory/TTTchoiceMem/.venv-rmbench-host512/bin/python
```

`configure.sh` 仅写入 Git 忽略的本地配置，保留已有文件。只训练时也可自行创建配置，只设 `JEPA_SHARED_ROOT`、`JEPA_TRAIN_ONLY=1` 和 BERT 来源；训练节点无需挂载仿真 checkout 或解释器。公共代码不包含当前开发机的 HDD 路径或 `runs` 链接。

在本地配置加入 `JEPA_BERT_SOURCE=/path/to/existing/bert-base-uncased`；必要时设置 `CONDA_EXE=/mnt/data/WORKSPACE/anaconda3/bin/conda`，然后运行 `bash scripts/a1008/host_setup.sh`。它建立独立 Python 3.10 policy 环境，固定 PyTorch 2.7.1 / CUDA 12.6 与核心依赖，运行 `pip check` 并保存依赖清单，不修改原仿真环境。需要其他兼容 wheel 时用 `JEPA_TORCH_INDEX` 明确指定；硬件预检仍须通过。

资产锁 [a1008-assets.json](../configs/a1008-assets.json) 保留官方 RMBench V1 full50、RGB codec、V-JEPA 2.1 ViT-L 权重与文本来源摘要。源数据校验仍以旧采样的 1,013 决策/81 块确认已发布数据身份；新版从 50 episodes、17,562 frames 计算 17,562 决策、最多 92 历史块。这两个统计对应不同采样，不能用旧计数限制新训练。

BERT-base-uncased 的五个本地文件按锁定摘要复制到 `$JEPA_SHARED_ROOT/weights/bert-base-uncased`，不改来源、不下载 T5。常驻主机在 CPU 缓存完整 prompt 的 128×768 token 特征与 mask；训练和 rollout 读缓存，不常驻 BERT，不做句向量池化或补零到 4096 维。BERT 与 T5 使用不同缓存后缀、来源和模型契约；`JEPA_TEXT_ENCODER=t5` 可显式选择原 T5 语言变体。

所有环境、数据、模型、日志、打包文件和 Inductor/Triton/CUDA 编译缓存均在 `JEPA_SHARED_ROOT`。沿用服务器短 IPC 路径 `JEPA_TMPDIR`（默认 `/tmp/jepa-$UID`），避免 DataLoader/planner UNIX socket 路径过长；该路径不存放权重、视觉特征或编译缓存。八卡默认每 rank 仅用一个 Inductor 编译线程（`TORCHINDUCTOR_COMPILE_THREADS=1`，可配置），避免初始编译过度抢占 CPU。入口尊重平台的 `CUDA_VISIBLE_DEVICES`、卡序号与 NCCL 设置，不套用 RTX 5090 的驱动/架构/仿真补丁。

## 缓存与存储

旧 v0 稀疏缓存不能提供新增逐帧决策和 off-grid 当前观测。新 BERT 密集缓存写入 `history_features_bert_v2/`，无损打包写入 `history_features_bert_v2_packed/`；文本缓存、原数据和视觉权重可以复用。视觉缓存始终保留每视角 14×14、FP32 特征；打包只改变布局，不改变 token、数值或未来监督。密集池化 v2 与全分辨率 v2_full 可复用同一冻结缓存，Predictor 是否池化由模型 checkpoint 契约决定。

入口在一个已分配 GPU 上以 `JEPA_CACHE_ENCODER_BATCH=4`、逐 episode 解码生成缓存，使用共享锁防止重复写入；未提交 manifest 的旧目录保留为独立未完成目录。打包校验源文件与每个 FP32 tensor 的摘要，已完成 episode 分片可续打包。原始 manifest 保持逻辑身份；正式训练同时接收原始和 packed manifest，校验编码器、数据、codec、代码、维度与来源。

完整密集 FP32 缓存约 197 GiB（观测 39.4 GiB、未来目标 157.6 GiB）；打包再增加约 197 GiB。节点按缺失缓存估算空间，并为 checkpoint 和原子替换预留 64 GiB；两份缓存均缺失时约需 458 GiB 空闲共享空间，另保留现有环境和权重占用。主机安装器的 120 GiB 阈值仅覆盖其准备阶段。空间不足可设 `JEPA_PACK_FEATURES=0` 使用原始读取器。

默认不强制预热共享页缓存。设 `JEPA_WARM_OBSERVATIONS=1` 时，入口在独立单进程完成 observation warmup 并确认成功退出，之后才启动 torchrun；分布式训练不携带 in-training warm flag。预热需要约 39.4 GiB 观测页和额外 16 GiB 可用内存，还须确认容器内存限额；页缓存可回收，不保证常驻。打包和预热在目标共享存储上的收益需实测。

## 节点预检与训练

入口按以下顺序执行，命令、日志与结果存入本次 `attempt-*`：

1. 校验运行计划和配方；记录 revision、工作区、依赖、设备、显存、拓扑、NCCL/allocator；执行实际八卡 all-reduce 并正常退出；Flex 后端还在单独 GPU 进程比较 FP32/BF16 前向和梯度、空行和跨 tile mask，通过后才准备大缓存。
2. 校验官方资产与 BERT 缓存，准备/校验密集视觉缓存、无损打包及可选单进程预热。
3. 按实际全量索引计算训练计划，确认无诊断子集、完整历史和数据来源。
4. 用正式模型、batch、precision、attention 和读取器做三次 AdamW 压力更新：前两次覆盖最长前缀相邻决策及不同可见性，最后一次覆盖最长混合 episode 批次。检查已有 AdamW/DDP 状态下的显存、有限 loss/梯度/参数、抽样 DDP 同步、冻结 Encoder、权重保存和正常退出。
5. 联合模式另查 native expert、示范回放、渲染和一个 32-step websocket 闭环；只训练模式明确记录跳过仿真。
6. 正式 From Scratch 训练或恢复同一新版实验完整状态；压力 checkpoint 仅作诊断。联合模式成功后释放八卡训练进程，再单卡串行做 20 次 native_qpos rollout；只训练模式将评测留给主机。

| 配置 | 新版默认 |
| --- | --- |
| 模型契约 | `v2_full_bert`，每视角历史 14×14，未来目标全分辨率 |
| 时间采样 | 完整历史间隔 4；每步训练决策，off-grid 当前叶子相互隔离 |
| 执行 | 每次执行 16 步后重新决策 |
| Attention | compiled FlexAttention，RoPE 复用，语言 padding 裁剪 |
| 计算复用 | microbatch 内可微历史前缀共享；不跨 optimizer update 保存 trainable KV |
| 显存控制 | 24 层 Predictor checkpointing、Action Expert checkpointing、紧凑 observation bank |
| Batch | 8 GPU × 每卡 2 × 累积 1 = 16 |
| 训练量 | 25 epochs × 1,097 updates = 27,425 updates |
| 每 epoch 样本 | 17,552；不足 global batch 的 10 个样本随确定性 shuffle 变化 |
| 保存 | 每 5 epochs 保存完整状态，保留最近两份权重及 5、10、20、最终 epoch 里程碑 |
| 优化器 | AdamW，LR 1e-4，weight decay 0.01，BF16 |

14×14 比 7×7 有四倍历史 token，增加计算和 KV 显存，参数量不会因此增加。40/80 GB A100 的容量、编译、通信和吞吐须在节点验证。若 batch 2 未通过，可用新运行名设置 `JEPA_PER_DEVICE_BATCH=1`，自动改为 accumulation=2，仍保持 global batch 16；此时没有跨样本共享历史的收益。调整后重新预检，不自动降低分辨率、截断历史或回退 eager dense attention。八卡耗时与 24 小时目标不能由本机或短压力测试推断。

## 进度、停止和完整恢复

默认运行目录 `$JEPA_SHARED_ROOT/runs/put_back_block_full50_v2_full_bert_ep25`。`status.json` 记录阶段/错误，`plan.json` 和 `recipe.json` 绑定计划与配方，每次启动生成 `attempt-*`。训练进度在 `train/progress.json`，最终状态在 `train/training_result.json`，完整断点为 `train/resume_state.pt`。运行锁拒绝两个节点同时写入。

```bash
cat /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_v2_full_bert_ep25/status.json
cat /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_v2_full_bert_ep25/train/progress.json
touch /mnt/data/dylan/d2/jepa_mem_assets/runs/put_back_block_full50_v2_full_bert_ep25/STOP
```

STOP 或 SIGTERM/SIGINT 请求在下一 optimizer 边界保存并退出，平台应留出至少五分钟宽限；SIGKILL/断电仅能恢复最近已提交断点，可能回退最多五个 epochs。删除自己创建的 STOP 后，在新八卡节点使用相同入口恢复。已完成时只训练模式校验结果后退出；联合模式可执行全新评测，不合并中断试验。

完整续训恢复权重、AdamW、scheduler、各 rank Python/NumPy/PyTorch/CUDA RNG、epoch 和数据游标；要求 world size、每卡 batch、累积、seed、数据来源、模型契约与总 schedule 相同。普通 `checkpoint_step*.pt` 是推理导出，不能视为完整断点。新采样不自动续接旧 v0/v0_bert，打包只有保持逻辑身份时才保留精确续训语义。

评测保持 whole-task instruction、native_qpos、原生步数上限和从 100000 开始的 20 次串行试验。500 步最多 126 历史块，14×14 常驻 BF16 KV 约 6.8 GiB、扩容预算约 14.3 GiB，另需模型与工作显存；容量耗尽报错，不淘汰历史。只训练模式 `evaluation=deferred_to_host` 不代表仿真通过。

## 验证边界

本机 CPU 可验证 BERT 梯度/共享计算等价性、缓存打包和完整 optimizer 恢复，不能代替实际八卡 A100 的驱动、kernel 编译、NCCL、最长历史容量或共享盘吞吐。历史 [2026-10-22 记录](a1008_validation_20261022.md) 对应旧配方，不作为本次硬件验收。只有实际完成 native simulator rollout 才能报告成功率。
