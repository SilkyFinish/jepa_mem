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
    p.add_argument("--cache-backend",choices=("incremental","recompute"),default="incremental")
    p.add_argument("--memory-precision",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--max-history-blocks",type=int,default=5)
    p.add_argument("--max-cache-bytes",type=int,default=1024**3)
    p.add_argument("--max-sessions",type=int,default=1)
    p.add_argument("--max-requests",type=int,default=4096)
    a=p.parse_args();logging.basicConfig(level=logging.INFO)
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    from vjepa_policy.memory_config import contract_from_checkpoint_metadata
    checkpoint=torch.load(a.ckpt,map_location="cpu",weights_only=False,mmap=True)
    contract=contract_from_checkpoint_metadata(checkpoint)
    del checkpoint
    if contract is None:
        policy=RMBenchPolicyServing(a.ckpt,a.pretrained_encoder,a.text_cache_dir,a.dataset_stats,a.device)
        metadata={"action_dim":14,"action_contract":"aloha_absolute_joint_14"}
    else:
        from vjepa_policy.policy_serving.memory import MemoryPolicyServing
        policy=MemoryPolicyServing.from_checkpoint(
            a.ckpt,a.pretrained_encoder,a.text_cache_dir,a.dataset_stats,a.device,
            backend=a.cache_backend,precision=a.memory_precision,
            max_history_blocks=a.max_history_blocks,max_cache_bytes=a.max_cache_bytes,
            max_sessions=a.max_sessions,max_requests=a.max_requests,seed=a.seed)
        metadata=policy.metadata
    WebsocketPolicyServer(policy=policy,host=a.host,port=a.port,metadata=metadata).serve_forever()
