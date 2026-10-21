import importlib.util
import unittest
# These numerical contracts require the installed policy environment.
# Missing optional packages skip this module; actual import errors still fail.
_REQUIRED_PACKAGES = ('numpy', 'torch', 'lerobot', 'vjepa_policy')
_missing = [name for name in _REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
if _missing:
    raise unittest.SkipTest(
        "Requires policy environment; missing: " + ", ".join(_missing)
    )

import numpy as np
import torch
from vjepa_policy.policy_serving.observations import (
    camera_observation, instruction, state_observation, to_uint8_hwc,
)


class ServingObservations(unittest.TestCase):
    def test_openpi_two_camera_history_is_preserved(self):
        now = np.zeros((2, 4, 3), dtype=np.uint8)
        now[..., 0] = 137
        past = np.full_like(now, 51)
        wire = {"observation/image": now, "observation/image_past": past,
                "observation/wrist_image": now+1, "observation/wrist_image_past": past+1,
                "observation/state": np.arange(8, dtype=np.float32), "prompt": "open the drawer"}
        np.testing.assert_array_equal(to_uint8_hwc(camera_observation(wire, "observation.images.image")), now)
        np.testing.assert_array_equal(to_uint8_hwc(camera_observation(wire, "observation.images.wrist_image", past=True)), past+1)
        np.testing.assert_array_equal(state_observation(wire), np.arange(8))
        self.assertEqual(instruction(wire), "open the drawer")

    def test_canonical_normalized_chw_matches_wire_uint8(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        canonical = torch.from_numpy(image).permute(2, 0, 1).float()/255
        np.testing.assert_array_equal(to_uint8_hwc(canonical.unsqueeze(0)), image)
        self.assertEqual(instruction({"task": ["open the drawer"]}), "open the drawer")

    def test_missing_past_frame_is_not_synthesized(self):
        with self.assertRaises(KeyError):
            camera_observation({"observation/image": np.zeros((2,4,3),dtype=np.uint8)},
                               "observation.images.image", past=True)

    def test_invalid_images_and_instruction_fail(self):
        for image in (np.zeros((4,4)), np.full((2,4,3), np.nan), np.full((2,4,3), 255.0)):
            with self.assertRaises(ValueError):
                to_uint8_hwc(image)
        with self.assertRaises(ValueError):
            instruction({"prompt": ""})


if __name__ == "__main__":
    unittest.main()
