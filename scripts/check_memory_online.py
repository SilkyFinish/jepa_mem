"""Verify online/recompute parity and real WebSocket requests on native RGB."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import time
from urllib.request import urlopen

import h5py
import numpy as np
import torch

from vjepa_policy.datasets.rmbench import CAMERAS, image_decoder, read_joint_vector
from vjepa_policy.policy_serving.memory import MemoryPolicyServing, PROTOCOL
from vjepa_policy.policy_serving.websocket import WebsocketPolicyServer


def checked(response):
    if not response.get("ok"):
        raise RuntimeError(response)
    return response


def reset_request(policy, instruction, sid="diagnostic"):
    return dict(op="reset", session_id=sid, request_id="reset", episode_id="native-replay",
                instruction=instruction, protocol=PROTOCOL, sampling_config=policy.config.to_dict())


def read_blocks(manifest_path, episode_index, config):
    manifest = json.loads(Path(manifest_path).read_text())
    row = manifest["episodes"][episode_index]
    decode = image_decoder(manifest["codec_path"])
    blocks = []
    with h5py.File(row["path"], "r") as handle:
        states = read_joint_vector(handle, "state")
        for index in range(6):
            end = index * 4
            frames = config.observation_frames(end)
            blocks.append(dict(block_id=index, end_step=end, frame_indices=list(frames),
                               images={cam: np.stack([decode(handle[f"vision/{cam}/colors"][frame])
                                                       for frame in frames]) for cam in CAMERAS},
                               state=states[end].copy()))
    return row, blocks


def compare_policies(online, recompute, row, blocks):
    policies = (online, recompute)
    for policy in policies:
        checked(policy.infer(reset_request(policy, row["instruction"])))
    results = []
    for number, selected in enumerate((blocks[:1], blocks[1:5])):
        elapsed = []
        for policy in policies:
            start = time.perf_counter()
            checked(policy.infer(dict(op="append", session_id="diagnostic", request_id=f"append-{number}",
                                      base_revision=number, blocks=selected)))
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            elapsed.append(time.perf_counter() - start)
        sessions = [policy.sessions["diagnostic"] for policy in policies]
        histories = [session.cache.snapshot() for session in sessions]
        step = selected[-1]["end_step"]
        error = lambda x, y: float((x.float() - y.float()).abs().max())
        layer_errors = []
        atol, rtol = (2e-5, 2e-4) if online.precision == "fp32" else (.06, .06)
        for a, b in zip(histories[0].layers, histories[1].layers):
            torch.testing.assert_close(a.key, b.key, atol=atol, rtol=rtol)
            torch.testing.assert_close(a.value, b.value, atol=atol, rtol=rtol)
            layer_errors.append({"key": error(a.key, b.key), "value": error(a.value, b.value)})
        futures, actions = [], []
        noise = torch.randn((1, 32, 14), device=online.device,
                            generator=torch.Generator(device=online.device).manual_seed(9))
        for policy, session, history in zip(policies, sessions, histories):
            decision = torch.tensor([step], device=policy.device)
            with torch.no_grad(), policy._precision():
                futures.append(policy.model.predictor.query_future(
                    history, session.cache.records[-1].state, session.language, session.language_mask,
                    decision, policy.model._future_layout(decision)))
                actions.append(policy.model.predict_from_history(
                    history, session.cache.records[-1].state, session.language, session.language_mask,
                    decision, initial_noise=noise))
        torch.testing.assert_close(futures[0].prediction, futures[1].prediction, atol=atol, rtol=rtol)
        torch.testing.assert_close(actions[0], actions[1], atol=atol, rtol=rtol)
        future_layers = []
        for a, b in zip(futures[0].layers, futures[1].layers):
            torch.testing.assert_close(a.key, b.key, atol=atol, rtol=rtol)
            torch.testing.assert_close(a.value, b.value, atol=atol, rtol=rtol)
            future_layers.append({"key": error(a.key, b.key), "value": error(a.value, b.value)})
        request = dict(op="infer", session_id="diagnostic", request_id=f"infer-{number}",
                       expected_revision=number+1, decision_step=step)
        physical = [checked(policy.infer(request))["actions"] for policy in policies]
        np.testing.assert_allclose(physical[0], physical[1], atol=atol, rtol=rtol)
        np.testing.assert_array_equal(physical[0], checked(online.infer(request))["actions"])
        assert sessions[0].cache.snapshot() is histories[0]
        results.append(dict(decision_step=step, valid_tokens=sessions[0].cache.valid_length,
                            layer_kv_max_error=layer_errors, future_kv_max_error=future_layers,
                            future_prediction_max_error=error(futures[0].prediction, futures[1].prediction),
                            normalized_action_max_error=error(actions[0], actions[1]),
                            physical_action_max_error=float(np.max(np.abs(physical[0]-physical[1]))),
                            append_seconds={"incremental": elapsed[0], "recompute": elapsed[1]}))
    for policy in policies:
        checked(policy.infer({"op": "close", "session_id": "diagnostic"}))
    return results


async def check_wire(policy, row, blocks):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from websockets.asyncio.server import serve
    server = WebsocketPolicyServer(policy, metadata=policy.metadata)
    async with serve(server.handle, "127.0.0.1", 0, process_request=server.health,
                     compression=None, max_size=None) as listener:
        port = listener.sockets[0].getsockname()[1]

        def exchange():
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                assert response.read() == b"ready\n"
            client = WebsocketClientPolicy("127.0.0.1", port)
            try:
                assert client.get_server_metadata()["protocol"] == PROTOCOL
                checked(client.infer(reset_request(policy, row["instruction"], sid="wire")))
                append = dict(op="append", session_id="wire", request_id="append",
                              base_revision=0, blocks=blocks[:5])
                ack = checked(client.infer(append))
                assert ack == checked(client.infer(append))
                request = dict(op="infer", session_id="wire", request_id="infer",
                               expected_revision=1, decision_step=16)
                response = checked(client.infer(request))
                repeated = checked(client.infer(request))
                np.testing.assert_array_equal(response["actions"], repeated["actions"])
                assert response["actions"].shape == (32, 14) and np.isfinite(response["actions"]).all()
                oversized = client.infer(dict(op="append", session_id="wire", request_id="over",
                                             base_revision=1, blocks=blocks[5:]))
                assert oversized["error"]["code"] == "history_capacity_exceeded"
                assert oversized["cache_revision"] == 1
                conflict = client.infer(dict(request, decision_step=0))
                assert conflict["error"]["code"] == "request_conflict"
                checked(client.infer({"op": "close", "session_id": "wire"}))
                checked(client.infer(reset_request(policy, row["instruction"], sid="disconnect")))
                return {"health": True, "action_shape": [32, 14], "finite_actions": True,
                        "exact_retry": True, "capacity_error_preserves_revision": True,
                        "request_conflict": True, "close": True}
            finally:
                client._ws.close()
        report = await asyncio.to_thread(exchange)
    # Exiting serve waits for handler finally/on_disconnect.
    assert not policy.sessions
    return dict(report, disconnect_cleanup=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "pretrained-encoder", "text-cache-dir", "manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset-stats")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"status": "failed", "diagnostic_only": True}
    try:
        online = MemoryPolicyServing.from_checkpoint(
            args.checkpoint, args.pretrained_encoder, args.text_cache_dir,
            args.dataset_stats, args.device, precision=args.precision)
        recompute = MemoryPolicyServing(online.model, online.normalizer, online.text_loader,
                                        backend="recompute", precision=args.precision)
        row, blocks = read_blocks(args.manifest, args.episode_index, online.config)
        report["parity"] = compare_policies(online, recompute, row, blocks)
        report["websocket"] = asyncio.run(check_wire(online, row, blocks))
        report["precision"] = args.precision
        report["source_episode"] = row["source_episode"]
        if online.device.type == "cuda":
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(online.device) / 1024**3
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
