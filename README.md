# V-JEPA Policy

<p align="center">
  <strong>Building Effective World-Action Models on Predictive Visual Latents</strong>
</p>

<p align="center">
  <a href="https://github.com/breez3young/VJEPA-Policy/blob/release/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2f6f68?style=flat" alt="MIT license"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="https://arxiv.org/abs/2609.37250"><img src="https://img.shields.io/badge/arXiv-2609.37250-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv:2609.37250"></a>
</p>

V-JEPA Policy builds a world-action model on the predictive latent space of a frozen **V-JEPA 2.1** encoder. An instruction-conditioned future-latent predictor and a flow-matching action expert couple future visual-state prediction with action generation. The predictor's future-informed context key–value states condition the action expert. The default model uses V-JEPA 2.1 ViT-L and a frozen T5-XXL text encoder with cached instruction embeddings.

<p align="center">
  <a href="assets/figures/architecture.pdf">
    <img src="assets/figures/architecture.png" width="1000" alt="V-JEPA Policy architecture: frozen visual and text encoders, an instruction-conditioned future predictor, and a flow-matching action expert.">
  </a>
</p>

**Architecture.** The future predictor learns from frozen visual targets, while its future-informed context states condition the action expert. Future clips provide supervision during training.

We study two variants:

- **From Scratch:** jointly learn the future predictor and action expert from task-specific demonstrations in a single downstream stage.
- **Pretrained Predictor:** first pretrain only the future predictor on DROID video–instruction pairs, with proprioceptive conditioning and no action labels or action expert. Transfer the predictor weights to downstream joint training with a freshly initialized action expert.

The visual and text encoders remain frozen in both variants.

## Current workflows

This checkout supports the native RMBench short-context baseline, Full-History v0 memory policy, generic dataset/encoder adapters, DROID predictor-only pretraining and RoboCasa-GR1.

## Installation

Use Python 3.10+ in a dedicated policy environment and install a GPU-compatible PyTorch build, then run `python -m pip install -e '.[serving]'` from the repository root. Keep policy and simulator environments separate. See [Setup](docs/setup.md).

## Training and evaluation

Follow [the native RMBench guide](examples/rmbench/README.md) for three-camera ALOHA data, instruction caching, short-context training and serial simulator rollouts. Keep each checkpoint with its statistics and frozen encoder/text-cache provenance.

For memory, follow [M0](docs/full_history_v0_m0.md), [M3 training](docs/full_history_v0_m3.md), [M4 online sessions](docs/full_history_v0_m4.md) and [attention measurements](docs/full_history_attention.md). Capacity and numerical diagnostics have their own validation limits.

The generic training entry point is `scripts/train_vjepa_policy.py`; [Training](docs/training.md) covers DROID predictor-only pretraining and resumption. RoboCasa-GR1 has separate [training](scripts/gr1/README.md) and [evaluation](examples/gr1/README.md) guides.

## Development

[AGENTS.md](AGENTS.md) maps training, serving, RMBench and GR-1 to the local skills. Shared checkpoint loading lives in `src/vjepa_policy/policy_serving/base.py`; each benchmark retains its own action decoder.

Run `python -m unittest discover -s tests -v` in the policy environment. See [Development checks](docs/development.md) for optional dependencies and GPU diagnostics.

## Repository map

```text
src/vjepa_policy/    encoders, datasets, models, training and serving
scripts/            generic training, DROID caching, memory diagnostics, GR-1 preparation
examples/rmbench/   native RMBench baseline and memory rollouts
examples/gr1/       RoboCasa-GR1 serving and evaluation
configs/            local artifact paths and shared recipe defaults
tests/              model, memory, serving and training contracts
docs/               setup, training and full-history implementation records
.agents/skills/     task-specific instructions
```

## Citation

If you find V-JEPA Policy useful for your research, please consider citing:

```bibtex
@misc{zhang2026vjepapolicybuildingeffective,
  title={V-JEPA Policy: Building Effective World-Action Models on Predictive Visual Latents},
  author={Yang Zhang and Jiangyuan Zhao and Chenyou Fan and Jiayu Hu and Xiu Yuan and Chenjia Bai and Xiu Li},
  year={2026},
  eprint={2609.37250},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37250},
}
```

## License

This repository is released under the [MIT License](LICENSE). Upstream visual encoders, T5 checkpoints, datasets, simulator assets, and evaluation harnesses retain their own licenses and terms.
