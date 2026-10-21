"""Validate the v0 recipe and native RMBench inputs, without training a model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_contract(path):
    # Keep contract-only checks usable in the source-only Python environment.
    spec = importlib.util.spec_from_file_location(
        "memory_v0_config", ROOT / "src/vjepa_policy/memory_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FullHistoryV0Config.from_dict(json.loads(Path(path).read_text()))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_file(value, manifest):
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def audit_manifest(config, manifest_path, output, controller, decode_all):
    import h5py
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    # Read the selected checkout even when the shared environment has an old
    # editable installation. Never reinstall or mutate that shared environment.
    sys.path.insert(0, str(ROOT / "src"))
    from vjepa_policy.datasets import VideoClipTransform
    from vjepa_policy.datasets.rmbench import CAMERAS, image_decoder, read_joint_vector

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    require(manifest["task"] == "put_back_block", "M0 pilot is put_back_block")
    require(manifest["action_contract"] == config.action_contract, "Action contract differs")
    require(tuple(CAMERAS) == config.cameras, "Legacy/v0 camera ordering differs")
    rows = manifest["episodes"]
    ids = [row["source_episode"] for row in rows]
    paths = [str(resolve_file(row["path"], manifest_path).resolve()) for row in rows]
    require(bool(rows) and len(ids) == len(set(ids)), "Empty/duplicate episode IDs")
    require(len(paths) == len(set(paths)), "Duplicate episode files")
    instructions = {row["instruction"] for row in rows}
    require(len(instructions) == 1 and all(text.strip() for text in instructions),
            "Pilot requires one fixed whole-task instruction")
    split = manifest["split"]
    require(split in ("full50", "diagnostic_train40", "diagnostic_heldout10"), "Unknown split")
    require(len(rows) == {"full50": 50, "diagnostic_train40": 40,
                         "diagnostic_heldout10": 10}[split], "Split size differs")
    codec = resolve_file(manifest["codec_path"], manifest_path)
    require(file_hash(codec) == manifest["codec_sha256"], "RGB codec changed")
    stats_path = resolve_file(manifest["statistics"], manifest_path)
    stats = json.loads(stats_path.read_text())
    metadata = stats["metadata"]
    require(metadata["action_contract"] == config.action_contract, "Statistics action contract")
    require(metadata["camera_order"] == list(config.cameras), "Statistics camera order")
    require(metadata["target_alignment"] == "action[t] = state[t+1]", "Action alignment metadata")
    require(metadata["gripper_indices"] == list(config.gripper_indices), "Gripper ordering")
    require(metadata["gripper_open"] == 1 and metadata["joint_units"] == "radians",
            "Joint/gripper units")
    require(metadata["rgb_decoder_sha256"] == manifest["codec_sha256"], "Statistics codec")
    require(metadata["image_transform"] == "stretch224_rgb", "Image transform differs")
    require(stats["normalization"] == {"action": "QUANTILE", "state": "QUANTILE",
                                      "clip_values": False}, "Normalization recipe differs")
    sources = metadata["statistics_source_episodes"]
    require(len(sources) == len(set(sources)), "Duplicate statistics episodes")
    if split == "diagnostic_heldout10":
        require(len(sources) == 40 and set(sources).isdisjoint(ids),
                "Heldout statistics leak validation episodes")
    else:
        require(set(sources) == set(ids), "Statistics are not from this training split")
    for key in ("action", "observation.state"):
        require(stats["features"][key]["shape"] == [14], "Statistics dimension differs")
        for name in ("min", "max", "mean", "std", "q01", "q99"):
            array = np.asarray(stats["stats"][key][name])
            require(array.shape == (14,) and np.isfinite(array).all(), "Invalid statistics")

    decode = image_decoder(str(codec))
    transform = VideoClipTransform((config.image_size, config.image_size))
    require(tuple(transform.mean.flatten().tolist()) == config.image_mean, "RGB mean differs")
    require(tuple(transform.std.flatten().tolist()) == config.image_std, "RGB std differs")
    torch.set_num_threads(1)
    report_rows, previews = [], []
    for row, path in zip(rows, paths):
        require(file_hash(path) == row["sha256"], f"Source hash changed: {path}")
        frames = row["frames"]
        decisions = config.decision_steps(frames)
        require(bool(decisions), f"No complete v0 decision in {path}")
        for step in decisions:
            config.sample_indices(str(row["source_episode"]), frames, step)
        max_blocks = len(config.observed_steps(decisions[-1]))
        decode_indices = tuple(range(frames)) if decode_all else tuple(
            range(0, frames, config.history_interval_steps)
        )
        with h5py.File(path, "r") as handle:
            require(handle["data_format_version"].asstr()[()] == "v1.0", "Data version differs")
            require(int(handle["additional_info/frequency"][()]) == 15, "Data cadence differs")
            state = read_joint_vector(handle, "state")
            action = read_joint_vector(handle, "action")
            require(state.shape == action.shape == (frames, 14), "State/action geometry differs")
            require(np.isfinite(state).all() and np.isfinite(action).all(), "Nonfinite vectors")
            alignment = float(np.abs(action[:-1] - state[1:]).max())
            require(alignment <= 1e-5, "Native next-state action alignment changed")
            for values in (state, action):
                grips = values[:, list(config.gripper_indices)]
                require((grips >= -1e-5).all() and (grips <= 1 + 1e-5).all(),
                        "Gripper convention differs")
            for camera in config.cameras:
                images = handle[f"vision/{camera}/colors"]
                require(len(images) == frames, "Camera/state lengths differ")
                for step in decode_indices:
                    rgb = decode(images[step])
                    require(rgb.dtype == np.uint8 and rgb.shape == (240, 320, 3),
                            f"Invalid RGB at {path}:{camera}:{step}")
                    if step == 0 and camera == "cam_head" and len(previews) < 8:
                        # Display exactly the existing model's transform,
                        # undoing normalization only for the preview image.
                        clip = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
                        transformed = transform(clip)
                        previews.append((row["source_episode"], transformed))
        report_rows.append({
            "source_episode": row["source_episode"], "frames": frames,
            "decision_samples": len(decisions), "last_decision_step": decisions[-1],
            "max_training_history_blocks": max_blocks,
            "max_training_history_tokens": max_blocks * config.tokens_per_block,
            "decoded_frames_per_camera": len(decode_indices),
            "next_state_action_max_error": alignment,
            "source_sha256_verified": True,
        })
        (output / "progress.json").write_text(json.dumps(report_rows, indent=2) + "\n")
        print(f"verified episode={row['source_episode']} frames={frames} "
              f"decisions={len(decisions)} max_history_blocks={max_blocks}", flush=True)

    if previews:
        # VideoClipTransform returns T,C,H,W with the shared 0.5/0.5
        # normalization. Undo only normalization for visual inspection.
        mean = transform.mean[0]
        std = transform.std[0]
        sheet = Image.new("RGB", (4 * 224, ((len(previews) + 3) // 4) * 248), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (episode, tensor) in enumerate(previews):
            restored = ((tensor[0] * std + mean).clamp(0, 1) * 255).round().byte()
            image = Image.fromarray(restored.permute(1, 2, 0).numpy())
            x, y = (index % 4) * 224, (index // 4) * 248
            sheet.paste(image, (x, y + 24))
            draw.text((x + 4, y + 4), f"episode {episode}, observed t=0", fill="black")
        sheet.save(output / "initial_cue_preview.png")

    return {
        "status": "passed", "scope": "M0 input contract; no memory model executed",
        "policy_arch": config.policy_arch, "model_version": config.model_version,
        "contract_sha256": config.fingerprint(), "task": manifest["task"], "split": split,
        "controller": controller, "controller_executed_this_check": False,
        "manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
        "statistics_sha256": file_hash(stats_path), "codec_sha256": file_hash(codec),
        "episodes": len(rows), "total_frames": sum(row["frames"] for row in rows),
        "total_decision_samples": sum(row["decision_samples"] for row in report_rows),
        "max_training_history_blocks": max(row["max_training_history_blocks"] for row in report_rows),
        "rgb_coverage": "all_frames" if decode_all else "all_v0_sampled_frames",
        "cue_visibility": "preview_exported_requires_visual_review",
        "records": report_rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(ROOT / "configs/memory_v0/input_contract.json"))
    parser.add_argument("--manifest", help="Prepared native RMBench manifest; optional")
    parser.add_argument("--output", help="Fresh report directory; required with --manifest")
    parser.add_argument("--controller", choices=("native_qpos", "dense15"), default="native_qpos")
    parser.add_argument("--decode-all", action="store_true", help="Also decode nonsampled frames")
    args = parser.parse_args(argv)
    config = load_contract(args.contract)
    if args.manifest is None:
        if args.output or args.decode_all:
            parser.error("--output/--decode-all require --manifest")
        print(json.dumps({"status": "contract_valid", "contract_sha256": config.fingerprint(),
                          "tokens_per_block": config.tokens_per_block,
                          "future_tokens": config.future_tokens,
                          "packed_state_dim": config.packed_state_dim}, indent=2))
        return
    if not args.output:
        parser.error("--manifest requires --output")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = audit_manifest(config, args.manifest, output, args.controller, args.decode_all)
    except Exception as error:
        (output / "results.json").write_text(json.dumps({
            "status": "failed", "error": str(error), "error_type": type(error).__name__,
        }, indent=2) + "\n")
        raise
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
