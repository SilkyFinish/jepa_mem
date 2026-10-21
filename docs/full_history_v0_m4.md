# Full-History v0 M4：在线缓存

## 当前状态（2026-10-21）

M4 的核心代码已落盘，短完整前缀的数值一致性和真实 WebSocket 请求已通过一次验证。**当前状态为“核心实现已完成，最终验收待补”**。按用户要求，尚未完成的测试、检验和最终审查暂时搁置，本次只整理文档，不继续启动验证任务。

已通过的 106 项测试和 GPU 验证发生在最后一轮边界修补之前。之后修改了客户端清理异常处理、失效 session 的重试拒绝、初始双帧一致性检查，以及重试时禁止改变决策时刻/重置内容；这些修补尚未重新测试。不能把此前的通过结果当成当前全部代码已经验收通过。

本阶段继续使用 `full-history-v0` 分支，没有另建模型仓库副本。代码和本文档尚未在本轮提交或推送。输入契约沿用 [M0](full_history_v0_m0.md)，训练接入沿用 [M3](full_history_v0_m3.md)。

## M4 具体改变了什么

M2/M3 在每次决策时重新计算从 episode 起点到当前时刻的全部历史 H。M4 增加推理专用缓存：每个真实观测块只计算一次 Predictor 的各层 K/V，后续块读取这些缓存，未来 F 和动作 A 读取当前已提交的历史。训练仍然走可微的全量重算路径，session 缓存不进入训练 checkpoint。

例如，决策时刻 0 先写入历史块 0；执行 16 个控制步后，新收到截止时刻 4、8、12、16 的四个观测块。增量路径只计算这四块，保留旧块 0 的各层 K/V。随后查询 F 和生成动作时，读取全部五块历史，不把预测未来或生成动作写回历史。

| 文件 | 本阶段职责 |
| --- | --- |
| [memory_predictor.py](../src/vjepa_policy/models/memory_predictor.py) | 增加 `append_history`，复用旧层 K/V，仅返回新块的暂存 K/V；保留 `forward_history` 全量重算。 |
| [memory_layouts.py](../src/vjepa_policy/models/memory_layouts.py)、[memory_types.py](../src/vjepa_policy/models/memory_types.py) | 为增量块保留全局 block ID 和时间坐标，提供布局拼接。 |
| [history_cache.py](../src/vjepa_policy/models/history_cache.py) | 管理每层缓存、有效长度、容量、块元数据和 revision，统一提交。 |
| [memory.py](../src/vjepa_policy/policy_serving/memory.py) | 实现 `MemoryPolicyServing`：session、输入检查、真实 RGB 编码、增量/重算后端、动作反归一化及请求重试。 |
| [websocket.py](../src/vjepa_policy/policy_serving/websocket.py) | 增加连接归属和断连清理 hook，普通 policy 仍调用原来的 `infer`。 |
| [serve_policy.py](../examples/rmbench/serve_policy.py) | 根据 checkpoint 架构分派 legacy 或 full-history 服务。 |
| [memory_client.py](../src/vjepa_policy/policy_serving/memory_client.py) | 收集观测块、保留待确认请求和未确认数据、核对服务端回复。 |
| [run_memory_client.py](../examples/rmbench/run_memory_client.py) | 新 RMBench 客户端入口，复用环境和原生控制器，每 16 步决策一次。 |
| [test_memory_v0_online.py](../tests/test_memory_v0_online.py) | CPU 数值、缓存事务、session 和客户端重试测试。 |
| [check_memory_online.py](../scripts/check_memory_online.py) | 真实 checkpoint/RGB 的增量与重算对比，以及真实 WebSocket 请求诊断。 |

## 缓存与提交规则

每个 session 的 batch 固定为 1。每层 K/V 存储形状为 `[1, heads, capacity_tokens, head_dim]`，有效部分长度为已提交块数乘以每块 token 数；生产 v0 每块为 588 tokens。K 是经过 RoPE 的 key，V 未施加 RoPE。缓存随真实历史增长，不自动丢弃旧块。

`append_history` 要求 eval 模式，并在 no_grad 下运行。第 l 层的新 token 读取旧缓存 H_l 和本次追加中允许可见的新 token；块内双向、块间因果，且每块使用自己的状态 q_i。所有层先计算成 delta，期间不修改已提交前缀。

提交前检查全部层的形状、dtype、device、有限值和块布局。需要扩容时先分配所有层的新缓冲，再复制旧前缀与新 delta；无需扩容时只写入旧有效长度以后的空间。所有拷贝成功后统一发布新历史、块记录和 revision。CUDA 路径在发布前同步，避免提前确认尚未完成的拷贝。分配/拷贝失败不会推进旧有效长度；可能污染设备上下文的运行时错误会使 session 进入需要重建的状态。

session 保存 episode、固定指令摘要、配置签名、块内容摘要和各块状态。模型参数在服务生命周期内应保持固定。session 和缓存对象位于模型外，不新增可训练参数，不写入模型 `state_dict`。

`infer` 只读已提交 H，临时生成 F，再生成动作；不会增加历史长度或 revision。请求之间串行执行；当前服务使用同一把锁，也不并行处理不同 session 的模型计算。

## 协议与输入契约

继续使用现有 OpenPI msgpack WebSocket 传输，协议名为 `rmbench_memory_v0`。服务端在图像转换前根据 `op` 分派请求。各操作严格检查字段，不接受客户端提供 latent、未来监督或额外条件字段。

| 操作 | 请求字段 | 结果 |
| --- | --- | --- |
| `reset` | `op, session_id, request_id, episode_id, instruction, protocol, sampling_config` | 建立空 session，返回 schema 和 revision 0。 |
| `append` | `op, session_id, request_id, base_revision, blocks` | 原子追加一批真实观测，返回最新 block ID 和 revision。 |
| `infer` | `op, session_id, request_id, expected_revision, decision_step` | 返回有限的 `actions[32,14]` 和所用 revision。 |
| `close` | `op, session_id` | 释放缓存、语言特征和重试结果；允许重复关闭。 |

`sampling_config` 必须是 `FullHistoryV0Config().to_dict()` 的完整配置；`instruction` 是固定 whole-task 指令。实际 checkpoint loader 校验 Encoder 权重和配置、训练统计文件及对应 T5 缓存的来源。默认读取 checkpoint 同目录的 `dataset_stats.json`，也可显式传入匹配的统计文件。

每个 block 只包含以下字段：

| 字段 | 规则 |
| --- | --- |
| `block_id` | 从 0 开始连续递增的整数。 |
| `end_step` | 等于 `block_id * 4`，使用真实控制步编号。 |
| `frame_indices` | `[max(0, end_step-4), end_step]`。 |
| `images` | 三个键 `cam_head / cam_left_wrist / cam_right_wrist`，每个值为 uint8 `[2,240,320,3]` RGB。 |
| `state` | 截止时刻的原始 float32 `[14]` 关节状态，必须有限。 |

服务端按原有三视角顺序将 RGB stretch 到 224，使用既有归一化与冻结 Encoder；观测特征不额外施加 target LayerNorm。每块单独送入 Encoder，以匹配此前 batch=1 的编码对照。状态先做 QUANTILE 归一化，再打包为 96 维。输出反归一化为原生 14D 绝对关节命令，仅夹爪维度 6、13 限制到 [0,1]，不裁剪臂关节到 [-1,1]。

当前状态直接取最后一个已提交块；`decision_step` 必须等于它的截止时刻，且为 16 的整数倍。不能只推进时间而不提交对应观测。

同一 session 内，`request_id` 绑定操作内容及 revision。相同请求重传返回原回复，推理重试复用原动作，不重新抽噪声；同 ID 改内容会返回 `request_conflict`。新的 append 请求可以包含已经提交的块，但摘要必须一致。只有包含新块且提交成功的 append 才将 revision 加 1；一批多块也只加 1。缺块、乱序、同编号内容变化和过期 revision 均拒绝。

常见错误码包括 `revision_conflict`、`block_order_conflict`、`block_conflict`、`history_capacity_exceeded`、`session_capacity_exceeded`、`request_capacity_exceeded`、`session_owner_mismatch` 和 `session_needs_reset`。语义错误以结构化 `ok=false` 回复，包含 session/revision 以及 `error.code/message/needs_reset`。未知 session 的 revision 为 null。

session 绑定创建它的连接；断连时清理该连接的 session。重连需要 reset 并从 episode 起点重建历史，当前没有跨连接续接协议。客户端帮助类保留未确认请求，允许调用方在同一仍可用的连接上重试；rollout 入口遇到异常会终止，不实现自动网络恢复。

## 客户端采样流程

客户端每执行一个控制步就读取观测；每 4 步组建一个两帧块，初始块重复第 0 帧。采样器只需保留上一张网格采样帧，另保留全部未确认完整块，收到服务端有效确认后才移除待发块。不会为了限制队列长度而丢失未提交历史。

默认流程为 `reset → append(块0) → infer(t=0) → 执行16步 → append(块1…4) → infer(t=16)`。每次预测 32 步，只执行前 16 步后重规划。episode 结束或异常退出时尝试关闭 session 和环境，并记录结果；最后一轮对异常清理的修补尚待验证。

新入口默认 1 个 trial、最多 32 个控制步，明确标记为 diagnostic。它保留 native expert-solvable 种子筛选，测试种子起点为 100000；`native_qpos` 和 `dense15` 分开记录。旧 `run_client.py` 仍服务原版短上下文模型。

## 使用入口与参数

下面命令是按当前接口整理的后续使用说明，本次文档整理没有执行这些命令。先通过本机忽略配置提供 Python、Encoder、文本缓存、RMBench 和 checkpoint 路径，从仓库根目录运行；输出路径须使用新目录，端口须未被占用。

服务端按 checkpoint 选择模型架构：

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$POLICY_PYTHON" examples/rmbench/serve_policy.py \
  --ckpt "$CHECKPOINT" \
  --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
  --cache-backend incremental --memory-precision fp32 \
  --max-history-blocks 5 --max-sessions 1 --port 10010
```

原版 checkpoint 仍进入 `RMBenchPolicyServing`；full-history v0 进入 `MemoryPolicyServing`。未知或不完整的架构标记不会静默落回 legacy。服务端架构分派 CLI 尚待补充独立启动检验，此前真实网络验证直接构造了同一服务类和 WebSocket server。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--cache-backend` | `incremental` | 可选 `recompute` 作为同一完整前缀的对照后端。 |
| `--memory-precision` | `fp32` | 可选 BF16；目前真实端到端验证仅覆盖 FP32。 |
| `--max-history-blocks` | 5 | 每 session 最大块数，超限报错，不截断。 |
| `--max-cache-bytes` | 1073741824 | 每 session K/V 缓冲、扩容重叠和暂存 delta 的预算。 |
| `--max-sessions` | 1 | 最大活动 session 数。 |
| `--max-requests` | 4096 | 每 session 的回复记录上限，不自动淘汰旧记录。 |

K/V 预算不包含模型权重、Encoder 激活、attention 工作区、文本特征、布局和 CPU 回复等所有开销，也不是整个服务的显存硬上限。允许多个 session 时要另外考虑总资源。recompute 后端会额外保存冻结观测特征，以便重新计算完整 H；incremental 后端不保存这些特征。

客户端短流程示例：

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$RMBENCH_PYTHON" -m examples.rmbench.run_memory_client \
  --rmbench-root "$RMBENCH_ROOT" --port 10010 \
  --trials 1 --max-steps 32 --seed-start 100000 \
  --controller native_qpos --output "$M4_ROLLOUT_OUTPUT"
```

默认 32 步只需要在 t=0 和 t=16 发起决策，最多读取 5 块历史。若继续决策到 t=32，需要 9 块；当前容量及 mask 限制会拒绝。短流程结束不代表完整任务完成或获得正式成功率。

以后恢复真实数据诊断时，可运行：

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$POLICY_PYTHON" scripts/check_memory_online.py \
  --checkpoint "$CHECKPOINT" \
  --pretrained-encoder "$VJEPA2_ENCODER_CHECKPOINT" \
  --text-cache-dir "$TEXT_EMBEDDING_CACHE" \
  --manifest "$RMBENCH_PREPARED_DATA/manifest.json" \
  --device cuda --precision fp32 --output "$M4_CHECK_OUTPUT"
```

该工具加载实际 checkpoint 和原生 HDF5 RGB，在 0/16 两个决策点比较两条路径，并通过自动分配的本地空闲端口验证 WebSocket 请求。它不是模拟器闭环评测。

## 已有验证证据

以下结果均来自最后一轮边界修补之前的代码，保留为阶段证据。

| 项目 | 结果与边界 |
| --- | --- |
| 完整 policy 环境测试 | 106 项通过，其中 12 项为新增 M4 测试；包含原版兼容回归。 |
| CPU 增量数值 | 逐块和批量追加的各层历史 K/V、未来预测及动作与全量重算对齐；BF16 覆盖 CPU 历史 K/V 对比。 |
| 缓存事务 | 模拟多层扩容失败、未使用尾部拷贝失败、OOM、容量拒绝；检查旧前缀、有效长度和 revision。 |
| session 与客户端 | 重试回复、内容冲突、乱序/缺块、session 归属/清理、请求上限、collector 未确认数据保留。 |
| 真实 GPU 模型 | M3 step 2048 小模型，实际预训练 ViT-L、T5 缓存、episode 0 原生 RGB，FP32。 |
| 真实 WebSocket | health、有限 32×14 动作、完全相同的重试输出、容量错误不推进 revision、请求冲突、close、断连清理均通过。 |
| 格式检查 | 当时的 `git diff --check` 通过；最新修补后的检查暂未重跑。 |

真实 GPU 对比采用 1 块和 5 块完整前缀；五块时历史长度为 2940 tokens。最大绝对误差如下：

| 比较项 | t=0，1 块 | t=16，5 块 |
| --- | ---: | ---: |
| 历史各层 K/V | 0 | 1.91e-6 |
| 未来各层 K/V | 0 | 9.54e-7 |
| Future 预测 | 0 | 1.38e-5 |
| 归一化动作 | 0 | 2.98e-7 |
| 反归一化动作 | 0 | 3.58e-7 |

本次真实诊断使用 GPU 1 的剩余显存，PyTorch 峰值已分配显存约 1.35 GiB，没有因缺少完全空闲的 GPU 阻塞。该数字只对应诊断模型和短前缀，不代表正式宽深模型、全长历史或其他进程的占用。

记录中的追加耗时包含图像编码、校验和首次初始化等开销；未开展充分预热、重复采样或长期运行对比，不能据此宣称在线缓存已获得明确的吞吐或延迟收益。

本机证据位于忽略目录 `runs/memory_v0_m4/`：`cpu_online.log`、`policy_tests.log`、`gpu_command.json`、`real_fp32.log`、`real_fp32/arguments.json` 和 `real_fp32/result.json`。真实 checkpoint 位于此前 M3 的 `overfit_resumed/checkpoint_step002048.pt`，是 2 层/128 维 Predictor、64 维 Action Expert 的诊断权重，不是正式任务性能模型。

## 已搁置的检验与后续边界

按用户指示，以下工作暂不执行，不作为已完成项：

1. 最后一轮修补后的回归测试，包括客户端清理错误是否保留原始异常、失效 session 是否拒绝旧回复、初始重复帧检查、重试内容变化拒绝。
2. 新客户端完整控制循环的异常退出/部分结果保存检验，以及 policy/simulator 环境中的 CLI 独立启动检查。
3. 当前全部改动的最终代码审查、语法和格式检查、无训练依赖环境的轻量测试。
4. BF16 真实权重服务与网络请求验证；已有 CPU BF16 历史算子测试不能替代这一项。
5. 实际模拟器闭环 rollout；当前只有原生记录 RGB 的在线服务请求，没有本版本闭环成功率。

81 块完整历史、长序列 attention、正式模型容量/性能和 current/window/full 对照属于后续 M5。M4 仍使用密集 mask 参考后端，mask 上限为 16,777,216 个元素；即使提高 `max-history-blocks`，其他容量检查仍可能拒绝。不会自动切换到滑动窗口，也不能通过移除旧 KV 来获得干净的 window 消融。

M3 中 world 分支尚未充分拟合的诊断问题也仍存在。缓存一致性证明两种执行方式在已测条件下输出接近，不证明记忆能力、长历史收益或任务成功率。
