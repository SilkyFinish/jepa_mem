"""Convert canonical LeRobot or OpenPI wire observations at the policy boundary."""
import numpy as np
import torch


def camera_observation(observation, camera_key, *, past=False):
    suffix = "_past" if past else ""
    canonical = camera_key + suffix
    name = camera_key.removeprefix("observation.images.")
    wire = "observation/" + name + suffix
    if canonical in observation:
        return observation[canonical]
    if camera_key.startswith("observation.images.") and wire in observation:
        return observation[wire]
    raise KeyError(f"Missing {'past-frame ' if past else ''}view {canonical!r} (wire alias {wire!r})")


def to_uint8_hwc(image):
    tensor_input = isinstance(image, torch.Tensor)
    if tensor_input:
        image = image.detach().cpu()
        if image.dtype != torch.uint8:
            image = image.float()
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Image must be HWC or CHW RGB, got {image.shape}")
    if tensor_input and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    elif image.shape[-1] != 3:
        if image.shape[0] != 3:
            raise ValueError(f"Image must have three RGB channels, got {image.shape}")
        image = image.transpose(1, 2, 0)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Images must be uint8 [0,255] or floating point [0,1]")
    if not np.isfinite(image).all() or image.min() < -1e-5 or image.max() > 1+1e-5:
        raise ValueError("Floating point images must be finite and in [0,1]")
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8))


def instruction(observation):
    text = observation.get("task", observation.get("prompt"))
    if isinstance(text, (list, tuple)) and len(text) == 1:
        text = text[0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Observation requires a task string/list or an OpenPI prompt")
    return text.strip()


def state_observation(observation):
    for key in ("observation.state", "observation/state"):
        if key in observation:
            return observation[key]
    raise KeyError("Missing observation.state (wire alias observation/state)")
