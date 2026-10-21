# Full-History v0：M1 核心算子

M1 已完成，建立在 M0 的固定输入契约上。本阶段提供可复用的算子和接口，保持原版模型默认调用与参数命名；尚未接入完整 v0 模型、训练入口或有状态服务。

## 实现内容

| 文件 | 改动 |
| --- | --- |
| [memory_types.py](../src/vjepa_policy/models/memory_types.py) | 定义 TokenLayout、LayerKV、HistoryEncoding、FutureEncoding，检查形状、设备、类型和决策归属；KV 保留计算图 |
| [memory_masks.py](../src/vjepa_policy/models/memory_masks.py) | 历史块因果、未来查询、动作查询三种参考 mask；处理 padding 和每个样本各自的决策截止时间 |
| [backbone/rope.py](../src/vjepa_policy/models/backbone/rope.py) | 共享三轴 RoPE、维度切分及空间插值；支持显式浮点时间坐标，保留旧频率配对开关 |
| [backbone/modules.py](../src/vjepa_policy/models/backbone/modules.py) | RoPEAttention 增加显式坐标和只读 prefix KV；新路径处理全屏蔽 query，并在 eval 关闭 dropout |
| [wm_predictor.py](../src/vjepa_policy/models/wm_predictor.py) | WorldModelPredictorBlock 透传新的坐标、mask、prefix 参数；旧调用保持原行为 |

没有新增模型权重。旧调用不传新参数时继续使用 legacy 分支，保留其位置索引、attention 分支和 dropout 行为。原先的 `modules.rotate_queries_or_keys` 导入位置仍可使用。M0 的配置、指纹和数据契约未改动。

## 可见性约定

| Query | 可以读取的 keys | keys 拼接顺序 |
| --- | --- | --- |
| H_i | 同行、同 episode 的历史块 H_j，块序号与观测截止时间均不晚于 H_i；同块跨视图双向可见 | 完整历史，或 prefix + 当前块 |
| F_t | 截止时间 ≤ t 的历史，以及本次决策的全部有效 F_t | H、F |
| A_t | 截止时间 ≤ t 的历史、本次 F_t、当前动作 chunk 的全部有效 A_t | H、F、A |

mask 形状统一为 `[B, 1, N_query, N_key]`，bool，`True` 表示允许读取。动作 chunk 和未来查询组都不是逐 token 自回归。H 的函数只接受历史布局，F 的函数不接受动作布局，以接口边界隔离三条流。

首版每行只含一个 episode 和一次决策。不同 episode 放在不同 batch 行；调用方负责确保 history/future/action 来自对应的同一 episode，M1 不维护 session 身份。F/A 布局的 `end_step` 表示查询所依据的观测截止时间，必须与传入的 `decision_step[B]` 一致；预测目标时间仅放在 `rope_coords`。因此，将预测时间改得更远不会扩大可见历史。首版不支持同行混合多个 episode 或多个决策的 packed attention。

`TokenLayout.rope_coords` 是 `[B,N,3]` 的浮点 time/height/width；空间部分为插值前的原始 patch 坐标，时间遵守 M0 的控制帧规则：H 为 i/8，F 为 (t+8,16,24,32)/8，A_j 为 (t+j+1)/8。动作标签本身不再次 shift。相机身份保留在 `view_id`，后续由模型的 view embedding 处理。

密集参考 mask 默认最多 16,777,216 个元素，分配前检查；超限报错，不截断历史。这个上限限制的是参考 mask 的结果元素数，不是整个 attention 的显存上限。长历史后续需要块式后端。

## Prefix attention 的调用方式

```python
from vjepa_policy.models.memory_masks import history_attention_mask

output, new_key, new_value = attention(
    block_hidden,
    position_coords=block_layout.rope_coords,
    prefix_kv=prefix_layer_kv,  # LayerKV 或首次调用时 None
    attn_mask=history_attention_mask(block_layout, prefix_and_block_layout),
    H_patches=14,
    W_patches=14,
    return_kv=True,
)
```

仅对当前输入投影 Q/K/V，并旋转当前 Q/K；历史 K 已经旋转，直接拼入 attention。返回的 K/V 和 `kv_out` hook 都只包含当前 token，调用方负责拼接，不会重复返回 prefix。读取期间不修改、不 detach 历史 KV，因此后续查询的 loss 可以向历史分支反传。

新路径要求同时提供显式坐标和完整的 bool mask。旧位置索引 `mask` 与新坐标不可混用；prefix 缺少显式坐标、KV 形状/类型/设备不匹配、错误 mask 形状及 `is_causal=True` 都报错。普通三角 causal mask 无法表达同块双向可见性。

新路径 attention dropout 使用 `attn_drop`，eval 时为零；投影 dropout 使用 `proj_drop`。全屏蔽 query 在 attention 和输出投影之后清零，前后向均保持有限值。Predictor block 的 residual、cross-attention、FFN 仍会产生 padding 位置的隐藏值；M2 必须继续传递有效位并屏蔽这些位置，M3 的 loss 也要排除 padding。

共享 RoPE 在 64 维 head 中仍分成 20/20/20 维加 4 维未旋转部分。新路径角度至少使用 float32 计算，输出保留 Q/K dtype，支持 BF16/autocast；视觉与动作投影后的 Q/K 必须采用同一组频率配对和空间插值参数。旧频率配对不是标准正交旋转，不能要求它满足严格的共同时间平移不变性；该性质只在 corrected 配对下验证，本阶段不自动切换已有权重的设置。

## 验证

从仓库根目录，在已安装依赖的 policy 环境中运行：

```bash
python -m unittest discover -s tests -p 'test_memory_v0_operators.py' -v
python -m unittest discover -s tests -v
```

本次完整 policy 环境使用 PyTorch 2.7.1，CPU 执行 61 项测试全部通过，其中新增 20 项 M1 测试。标准库环境执行 36 项，32 项通过，4 个依赖数值环境的模块跳过。日志保存在本机被 Git 忽略的 `runs/memory_v0_m1/`。

覆盖范围：

- 从 M0 提交 `e9be66b` 冻结旧 attention 作为独立参考；32 组配对/后端/插值/位置选择/train-eval 配置下，输出和 K/V 逐元素一致；双向 strict load 通过。
- Predictor block 原调用与旧计算过程输出逐元素一致，参数名和形状不变。
- 三层小模型完整重算与逐块追加的输出、逐层 K/V 在 FP32/BF16 容差内一致；训练时输入梯度和参数梯度一致。
- 同块跨视图双向可见、后续历史扰动不影响较早输出、batch 行隔离、padding 隔离。
- F 查询不读取决策时刻之后的历史；合法历史有梯度，非法历史梯度为零；重复查询不修改 prefix。
- SDPA 与显式 attention 的输出和梯度一致；支持空 prefix、BF16 autocast；全屏蔽行输出与梯度为零且无 NaN。
- 显式坐标与旧位置索引一致；corrected RoPE 的共同时间平移测试通过；无效参数及超大参考 mask 被拒绝。

这些是算子级和小模型级验证，没有运行 GPU 训练、完整 v0 推理或模拟器评测，也不代表长序列显存/吞吐已经达标。冻结的旧实现仅用于测试，位于 [tests/fixtures/rope_attention_e9be66b.py](../tests/fixtures/rope_attention_e9be66b.py)。

## 下一步

M2 在这些接口上实现 history/future 两阶段 Predictor、逐历史块状态条件，以及 Action Expert 的共享 RoPE 和 H/F 上下文读取。M3 再接入 dataset、构建器、checkpoint 版本检查与训练；持久缓存生命周期后续单独实现。目前训练和服务默认仍进入原版模型。
