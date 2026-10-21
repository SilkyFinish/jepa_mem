"""Serve a prepared RMBench policy; actions remain absolute native joints."""
import argparse
import logging
import numpy as np
import torch
from vjepa_policy.policy_serving.rmbench import RMBenchPolicyServing
from vjepa_policy.policy_serving.websocket import WebsocketPolicyServer

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("ckpt","pretrained-encoder","text-cache-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--dataset-stats");p.add_argument("--device",default="cuda")
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=10010);p.add_argument("--seed",type=int,default=7)
    a=p.parse_args();logging.basicConfig(level=logging.INFO)
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    policy=RMBenchPolicyServing(a.ckpt,a.pretrained_encoder,a.text_cache_dir,a.dataset_stats,a.device)
    WebsocketPolicyServer(policy=policy,host=a.host,port=a.port,
                          metadata={"action_dim":14,"action_contract":"aloha_absolute_joint_14"}).serve_forever()
