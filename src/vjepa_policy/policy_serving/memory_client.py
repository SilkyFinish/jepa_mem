"""RMBench memory request construction; no predictor/encoder imports."""
import copy
import uuid

import numpy as np

from vjepa_policy.memory_config import FullHistoryV0Config


class HistoryCollector:
    """Retain one sampled frame plus unacknowledged complete observation blocks."""

    def __init__(self):
        self.config = FullHistoryV0Config()
        self.last_step = -1
        self.previous_images = None
        self.pending = []
        self.last_ack = -1

    def add(self, step, images, state):
        if type(step) is not int or step != self.last_step + 1:
            raise ValueError("collector requires each executed control step in order, starting at zero")
        if step % self.config.history_interval_steps:
            self.last_step = step
            return
        if set(images) != set(self.config.cameras):
            raise ValueError("three canonical cameras are required")
        current = {}
        for camera in self.config.cameras:
            pixels = np.asarray(images[camera])
            if pixels.dtype != np.uint8 or pixels.shape != self.config.source_image_shape:
                raise ValueError("collector images must be native uint8 RGB")
            current[camera] = pixels.copy()
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("collector state must be finite [14]")
        past = self.previous_images if self.previous_images is not None else current
        self.pending.append({
            "block_id": step // 4, "end_step": step,
            "frame_indices": list(self.config.observation_frames(step)),
            "images": {key: np.stack((past[key], current[key])) for key in self.config.cameras},
            "state": state.copy(),
        })
        self.previous_images = current
        self.last_step = step

    def completed_blocks(self):
        return copy.deepcopy(self.pending)

    def acknowledge(self, last_block_id):
        if type(last_block_id) is not int or not self.last_ack <= last_block_id <= self.last_step // 4:
            raise ValueError("invalid block acknowledgement")
        self.pending = [block for block in self.pending if block["block_id"] > last_block_id]
        self.last_ack = last_block_id


class MemorySessionClient:
    """Retain the exact outstanding request until it is acknowledged.

    A transport exception may be retried on the same live connection. Reconnect
    starts a new session and requires replay from episode zero.
    """

    def __init__(self, transport, session_id=None):
        self.transport = transport
        self.session_id = session_id or uuid.uuid4().hex
        self.revision = 0
        self.last_block_id = -1
        self.pending_request = None
        self.sequence = 0
        self.episode_id = None

    def _send(self, op, **fields):
        if self.pending_request is None:
            self.sequence += 1
            self.pending_request = {"op": op, "session_id": self.session_id,
                                    "request_id": str(self.sequence), **fields}
        elif self.pending_request["op"] != op:
            raise RuntimeError("retry or close the outstanding request before another operation")
        if op == "infer" and fields["decision_step"] != self.pending_request["decision_step"]:
            raise RuntimeError("cannot change decision_step while retrying an inference")
        if op == "reset" and any(fields[key] != self.pending_request[key] for key in fields):
            raise RuntimeError("cannot change episode/instruction while retrying reset")
        reply = self.transport.infer(self.pending_request)
        if reply.get("session_id") != self.session_id:
            raise ValueError("server replied for a different session")
        if not reply.get("ok"):
            raise RuntimeError(f"memory request failed: {reply.get('error')}")
        if self.episode_id is not None and reply.get("episode_id") != self.episode_id:
            raise ValueError("server replied for a different episode")
        if type(reply.get("cache_revision")) is not int or reply["cache_revision"] < self.revision:
            raise ValueError("server revision moved backwards")
        if type(reply.get("last_block_id")) is not int or reply["last_block_id"] < self.last_block_id:
            raise ValueError("server block acknowledgement moved backwards")
        expected = self.revision
        if op == "append" and any(b["block_id"] > self.last_block_id for b in self.pending_request["blocks"]):
            expected += 1
        if reply["cache_revision"] != expected:
            raise ValueError("unexpected server revision")
        if op == "append" and reply["last_block_id"] != max(self.last_block_id, self.pending_request["blocks"][-1]["block_id"]):
            raise ValueError("server did not acknowledge exactly the submitted prefix")
        if op == "infer":
            actions = np.asarray(reply.get("actions"))
            if (reply.get("decision_step") != self.pending_request["decision_step"]
                    or actions.shape != (32, 14) or not np.isfinite(actions).all()
                    or reply["last_block_id"] != self.last_block_id):
                raise ValueError("invalid inference acknowledgement")
        self.revision, self.last_block_id = reply["cache_revision"], reply["last_block_id"]
        self.pending_request = None
        return reply

    def reset(self, episode_id, instruction):
        self.episode_id = episode_id
        return self._send("reset", episode_id=episode_id, instruction=instruction,
                          protocol="rmbench_memory_v0", sampling_config=FullHistoryV0Config().to_dict())

    def append(self, collector):
        if not collector.pending and self.pending_request is None:
            return
        reply = self._send("append", base_revision=self.revision, blocks=collector.completed_blocks())
        collector.acknowledge(reply["last_block_id"])
        return reply

    def infer(self, decision_step):
        return self._send("infer", expected_revision=self.revision, decision_step=decision_step)["actions"]

    def close(self):
        reply = self.transport.infer({"op": "close", "session_id": self.session_id})
        if reply.get("session_id") != self.session_id or not reply.get("ok") or not reply.get("closed"):
            raise RuntimeError("session close failed")
        self.pending_request = None
        return reply
