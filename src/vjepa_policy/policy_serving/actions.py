"""Explicit LIBERO gripper conventions at the simulator boundary."""
import numpy as np


def resolve_gripper_convention(explicit=None, metadata=None):
    convention = explicit or (metadata or {}).get("gripper_action_convention", "lerobot")
    if convention not in ("lerobot", "robosuite"):
        raise ValueError(f"Unsupported gripper action convention: {convention!r}")
    return convention


def to_env_action(action, binarize_gripper=True, gripper_action_convention="lerobot"):
    """Decode unnormalized actions: LeRobot 0=closed/1=open; robosuite -1=open/+1=closed."""
    convention = resolve_gripper_convention(gripper_action_convention)
    action = np.asarray(action, dtype=np.float32).copy()
    if convention == "lerobot":
        action[..., -1] = 1.0 - 2.0 * action[..., -1]
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    else:
        action[..., -1] = np.clip(action[..., -1], -1.0, 1.0)
    action[..., :6] = np.clip(action[..., :6], -1.0, 1.0)
    return action
