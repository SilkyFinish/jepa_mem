"""Episode boundaries and native joint output are critical benchmark contracts."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

# These numerical contracts require the installed policy environment.
# Missing optional packages skip this module; actual import errors still fail.
_REQUIRED_PACKAGES = ('numpy', 'h5py', 'torch', 'lerobot', 'vjepa_policy')
_missing = [name for name in _REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
if _missing:
    raise unittest.SkipTest(
        "Requires policy environment; missing: " + ", ".join(_missing)
    )

import h5py
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from vjepa_policy.datasets import DEFAULT_PROMPT
from vjepa_policy.datasets.normalization import ActionStateNormalizer
from vjepa_policy.datasets.rmbench import CAMERAS, RMBenchDataset, decode_joint_actions

class RMBenchContractsTest(unittest.TestCase):
    def test_joint_outputs_preserve_radians_and_both_grippers(self):
        action=np.tile(np.arange(14,dtype=np.float32),(32,1))
        action[:,6]=-.2;action[:,13]=1.2
        decoded=decode_joint_actions(action)
        self.assertEqual(decoded[0,10],10.)
        self.assertEqual(decoded[0,6],0.)
        self.assertEqual(decoded[0,13],1.)
        self.assertEqual(action[0,6],np.float32(-.2))
        with self.assertRaises(ValueError):decode_joint_actions(np.zeros((32,7)))
        with self.assertRaises(ValueError):decode_joint_actions(np.full((32,14),np.nan))

    def test_episode_padding_keeps_action_and_observation_in_same_episode(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);codec=root/"codec.py"
            codec.write_text("def decode_image_bit(value):\n    return value\n")
            stats={key:{"min":np.zeros(14),"max":np.ones(14)*100} for key in ("action","observation.state")}
            normalizer=ActionStateNormalizer(
                features={"action":PolicyFeature(type=FeatureType.ACTION,shape=(14,)),
                          "observation.state":PolicyFeature(type=FeatureType.STATE,shape=(14,))},
                stats=stats,action_mode="IDENTITY",state_mode="IDENTITY")
            normalizer.save(root/"stats.json")
            rows=[]
            for episode,offset in enumerate((0,100)):
                p=root/f"{episode}.hdf5"
                with h5py.File(p,"w") as f:
                    for group in ("state","action"):
                        for field,width in (("left_arm_joint_states",6),("left_ee_joint_states",1),("right_arm_joint_states",6),("right_ee_joint_states",1)):
                            values=np.arange(5,dtype=np.float32)[:,None]+offset+(1 if group=="action" else 0)
                            f.create_dataset(f"{group}/{field}",data=np.repeat(values,width,axis=1))
                    for camera in CAMERAS:
                        rgb=np.repeat((np.arange(5,dtype=np.uint8)+offset)[:,None,None,None],2,axis=1)
                        rgb=np.repeat(np.repeat(rgb,2,axis=2),3,axis=3)
                        f.create_dataset(f"vision/{camera}/colors",data=rgb)
                rows.append({"path":str(p),"frames":5,"instruction":"return the block"})
            manifest={"action_contract":"aloha_absolute_joint_14","episodes":rows,"codec_path":str(codec),"statistics":str(root/"stats.json")}
            (root/"manifest.json").write_text(json.dumps(manifest))
            prompt=DEFAULT_PROMPT.format(task="return the block")
            torch.save({"context":torch.zeros(128,4096),"mask":torch.ones(128,dtype=torch.bool)},root/(hashlib.sha256(prompt.encode()).hexdigest()+".t5_len128.pt"))
            args=argparse.Namespace(views=["observation.images."+x for x in CAMERAS],view_layout="independent",
                action_dim=14,proprio_dim=14,num_frames=5,past_frames=2,video_frame_stride=2,crop_size=2,
                text_cache_dir=str(root),context_len=128,dataset_stats=None,output_dir=str(root/"output"),require_full_future_window=False)
            dataset=RMBenchDataset(root/"manifest.json",args)
            tail=dataset[4]
            self.assertEqual(tail["action"][:,0].tolist(),[5.,5.,5.,5.])
            self.assertEqual(tail["action_is_pad"].tolist(),[False,True,True,True])
            self.assertEqual(dataset[5]["proprio"][0].item(),100.)
            first_video=dataset[5]["video"]
            self.assertTrue(torch.equal(first_video[:,:,0],first_video[:,:,1]))
            with self.assertRaises(IndexError):dataset[len(dataset)]
            dataset.close()

if __name__=="__main__":unittest.main()
