"""CLI contract for the native RMBench Full-History v0 training route."""
from vjepa_policy.memory_config import FullHistoryV0Config

HISTORY_FACTORY = "vjepa_policy.datasets.rmbench_history:build_dataset"


def add_history_args(parser):
    parser.add_argument("--policy-arch", choices=("legacy", "full_history"), default="legacy")
    parser.add_argument("--text-encoder", choices=("t5", "bert"), default="t5",
                        help="Frozen text features; BERT selects the explicit v0_bert contract")
    parser.add_argument("--history-feature-cache", help="Frozen observation/target cache manifest")
    parser.add_argument("--history-episode-ids", nargs="+", help="Explicit diagnostic episode selection")
    parser.add_argument("--history-max-decision-step", type=int,
                        help="Explicit early-decision curriculum; prefixes remain complete")
    parser.add_argument("--max-history-blocks", type=int,
                        help="Reject a dataset exceeding this capacity; never truncate it")
    parser.add_argument("--history-bucket-size", type=int, default=128,
                        help="Sort shuffled pools by prefix length; 0 disables bucketing")
    parser.add_argument("--history-encoder-batch-size", type=int, default=4)
    parser.add_argument("--history-attention-backend", choices=("segmented", "dense"), default="segmented",
                        help="Exact query-block SDPA, or bounded dense correctness reference")
    return parser


def set_history_defaults(parser, text_encoder="t5"):
    config = FullHistoryV0Config.for_text_encoder(text_encoder)
    parser.set_defaults(
        encoder="vjepa2_1_vitl", encoder_family="vjepa2_1", model_name="vit_large",
        encoder_checkpoint_key="ema_encoder", view_layout="independent",
        views=["observation.images." + camera for camera in FullHistoryV0Config().cameras],
        dataset_factory=HISTORY_FACTORY, context_len=config.text_length, lang_dim=config.text_dim,
        action_dim=14, proprio_dim=14, max_state_dim=48, proprio_encoding="identity",
        require_full_future_window=True, batch_size=1,
    )


def validate_history_args(args):
    config = FullHistoryV0Config.for_text_encoder(getattr(args, "text_encoder", "t5"))
    fixed = dict(
        encoder=config.encoder, encoder_family="vjepa2_1", model_name="vit_large",
        encoder_checkpoint_key="ema_encoder", crop_size=224, patch_size=16, tubelet_size=2,
        num_frames=33, past_frames=4, video_frame_stride=4, context_tubelets=1,
        view_layout="independent", action_dim=14, proprio_dim=14, max_state_dim=48,
        proprio_encoding="identity", context_len=config.text_length, lang_dim=config.text_dim, action_chunk_size=32,
        action_normalization="QUANTILE", state_normalization="QUANTILE",
        clip_normalized=False, require_full_future_window=True, instruction_field="task",
        loss_exp=1.0,
    )
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            raise ValueError(f"Full-History v0 requires --{name.replace('_', '-')}={expected!r}")
    if list(args.views or ()) != ["observation.images." + c for c in config.cameras]:
        raise ValueError("Full-History v0 requires ordered head, left wrist, right wrist cameras")
    for name in ("action_indices", "state_indices", "relative_action_indices", "camera_key"):
        if getattr(args, name, None):
            raise ValueError(f"Full-History v0 does not support {name}")
    if args.dataset_factory != HISTORY_FACTORY:
        raise ValueError(f"Full-History v0 requires --dataset-factory {HISTORY_FACTORY}")
    if args.history_encoder_batch_size < 1 or args.history_bucket_size < 0:
        raise ValueError("Invalid history encoding batch size or bucket size")
    if args.max_history_blocks is not None and args.max_history_blocks < 1:
        raise ValueError("max-history-blocks must be positive")
    return config


def validate_predictor_transfer(checkpoint, predictor):
    """Shapes alone cannot establish the source's state/rotary semantics."""
    from vjepa_policy.memory_config import contract_from_checkpoint_metadata
    contract = contract_from_checkpoint_metadata(checkpoint)
    if contract is not None and contract.text_dim != predictor.lang_dim:
        raise ValueError("Predictor transfer text contract differs from the target model")
    topology = checkpoint.get("model_topology") or {}
    attention = predictor.predictor_blocks[0].self_attn
    required = {
        "packed_proprio_dim": 96,
        "predictor_rope_frequency_pairing": "corrected" if attention.corrected_frequency_pairing else "legacy",
        "predictor_interpolate_rope": bool(attention.interpolate_rope),
    }
    for key, value in required.items():
        if topology.get(key) != value:
            raise ValueError(f"Predictor transfer requires source {key}={value!r}; metadata missing or incompatible")


def validate_reference_capacity(dataset, batch_size):
    """Fail before training, rather than hitting a late decision mid-epoch."""
    from vjepa_policy.models.memory_masks import DEFAULT_MAX_MASK_ELEMENTS
    config = FullHistoryV0Config()
    history = max(dataset.history_lengths) * config.tokens_per_block
    future, action = config.future_tokens, config.action_horizon
    elements = batch_size * max(history * history, future * (history + future),
                                action * (history + future + action))
    if elements > DEFAULT_MAX_MASK_ELEMENTS:
        raise ValueError(
            f"Selected batch/prefix needs {elements} dense mask elements; M1/M2 reference "
            f"backend limit is {DEFAULT_MAX_MASK_ELEMENTS}. Use an explicit early-decision "
            "curriculum (--history-max-decision-step 16 with --batch-size 1), or implement "
            "a long-sequence attention backend. Histories will never be truncated."
        )
