"""Within-update shared-prefix equivalence and complete anchor coverage."""
import copy
import unittest
import torch
from test_memory_v0_model import batch, enable_attention_gates
from test_memory_v1_spatial import model
from vjepa_policy.memory_config import FullHistoryV2Config
from vjepa_policy.trainer import _EpochRandomSampler

def shared_batch():
    b = batch()
    # Four decisions with overlapping regular blocks and independent off-grid leaves.
    b = {k: v[:1].repeat(4,*([1]*(v.ndim-1))) if torch.is_tensor(v) else v for k,v in b.items()}
    base_z, base_state = b["history_latents"][0].clone(), b["history_state"][0].clone()
    for i,t in enumerate((12,13,14,16)):
        ends = FullHistoryV2Config().observed_steps(t)
        count = len(ends)
        b["history_valid"][i] = torch.arange(5) < count
        b["history_end_step"][i] = torch.tensor(list(ends)+[0]*(5-count))
        b["decision_step"][i] = t
        for j,e in enumerate(ends):
            if e % 4 == 0:
                b["history_latents"][i,j] = base_z[e//4]
                b["history_state"][i,j] = base_state[e//4]
            else:
                b["history_latents"][i,j] = base_z[0] + e*.01
                b["history_state"][i,j] = base_state[0] + e*.01
        b["current_state"][i] = b["history_state"][i,count-1]
    b["episode_id"] = ["0"]*4
    return b

class SharedHistoryTests(unittest.TestCase):
    def test_same_losses_gradients_and_no_future_leakage(self):
        torch.manual_seed(3)
        reference, candidate = model(),model(checkpointing=True)
        reference.config = candidate.config = FullHistoryV2Config()
        reference.predictor.trim_condition_padding = False
        reference.action_expert.trim_condition_padding = False
        enable_attention_gates(reference.action_expert)
        candidate.load_state_dict(reference.state_dict())
        candidate.share_history_prefixes = True
        candidate.action_expert.checkpoint_memory_blocks = True
        b = shared_batch()
        noise, time = torch.randn_like(b["action"]),torch.tensor([.1,.3,.5,.7])
        a = reference(b,flow_noise=noise,flow_time=time)[0]
        c = candidate(b,flow_noise=noise,flow_time=time)[0]
        torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-5)
        a.backward();c.backward()
        for (name,p),(_,q) in zip(reference.named_parameters(),candidate.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5,msg=name)
        # Changing future regular block 16 cannot affect decision 12's predicted action.
        candidate.train()
        before = candidate._encode_context(b)[1].prediction[0].detach()
        changed = copy.deepcopy(b)
        changed["history_latents"][3,-1] += 50
        after = candidate._encode_context(changed)[1].prediction[0].detach()
        torch.testing.assert_close(before,after,atol=2e-6,rtol=2e-5)

    def test_mixed_episode_checkpoint_preserves_batch_mean_and_gradients(self):
        torch.manual_seed(23)
        reference, candidate = model(), model(checkpointing=True)
        reference.config = candidate.config = FullHistoryV2Config()
        enable_attention_gates(reference.action_expert)
        candidate.load_state_dict(reference.state_dict())
        candidate.share_history_prefixes = True
        candidate.action_expert.checkpoint_memory_blocks = True
        b = shared_batch()
        b["episode_id"] = ["0", "1", "2", "3"]
        b["action_is_pad"][1, 7:] = True
        b["action_is_pad"][2] = True
        b["future_valid"][3, 12:] = False
        noise, time = torch.randn_like(b["action"]), torch.tensor([.1,.3,.5,.7])
        a, am = reference(b, flow_noise=noise, flow_time=time)
        c, cm = candidate(b, flow_noise=noise, flow_time=time)
        torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-5)
        for key in am:
            torch.testing.assert_close(am[key],cm[key],atol=2e-6,rtol=2e-5)
        a.backward(); c.backward()
        for (name,p),(_,q) in zip(reference.named_parameters(),candidate.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5,msg=name)

    def test_grouped_sampler_covers_every_anchor_and_resumes(self):
        class Data:
            share_history_prefixes = True
            index = [(e,t) for e,n in enumerate((17,22,19)) for t in range(n)]
            history_lengths = [t//4+1 for e,t in index]
            def __len__(self):return len(self.index)
        data=Data()
        a = _EpochRandomSampler(data,seed=7,batch_size=4,bucket_size=128)
        first=list(a)
        self.assertEqual(sorted(first),list(range(len(data))))
        self.assertEqual(first,list(_EpochRandomSampler(data,seed=7,batch_size=4,bucket_size=128)))
        a.set_position(12)
        self.assertEqual(list(a),first[12:])
        a.start_epoch(1)
        self.assertEqual(sorted(a),list(range(len(data))))
        self.assertNotEqual(list(a),first)
