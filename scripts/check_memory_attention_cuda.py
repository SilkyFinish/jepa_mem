"""CUDA output/gradient parity and actual fused-kernel check; no model downloads."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vjepa_policy.models.memory_attention import history_attention_plan
from vjepa_policy.models.memory_layouts import visual_layout
from vjepa_policy.models.memory_masks import history_attention_mask


def relative_error(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.manual_seed(71)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__, operator=[], model={})
    h = visual_layout(torch.arange(5, device="cuda")[None]*4,
                      torch.ones(1,5,device="cuda",dtype=torch.bool),
                      grid_height=14,grid_width=14,num_views=3)
    plan, mask = history_attention_plan(h), history_attention_mask(h)
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, .02)):
        tensors = [torch.randn(1,16,2940,64,device="cuda",dtype=dtype,requires_grad=True) for _ in range(3)]
        refs = [x.detach().float().requires_grad_() for x in tensors]
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(*refs, attn_mask=mask)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION if dtype == torch.bfloat16 else SDPBackend.MATH):
            actual = plan.attend(*tensors)
            actual.float().square().mean().backward()
        expected.square().mean().backward()
        errors = [relative_error(a.grad, b.grad) for a,b in zip(tensors,refs)]
        row = dict(dtype=str(dtype), output_relative_l2=relative_error(actual,expected),
                   output_max_abs=float((actual.float()-expected).abs().max()),
                   qkv_gradient_relative_l2=errors, tolerance_relative_l2=tolerance)
        assert row["output_relative_l2"] < tolerance and max(errors) < tolerance, row
        report["operator"].append(row)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        plan.attend(*[x.detach().requires_grad_() for x in tensors]).float().square().mean().backward()
    report["sdpa_events"] = [e.key for e in prof.key_averages() if "scaled_dot_product" in e.key]
    assert any("flash_attention" in name for name in report["sdpa_events"]), report["sdpa_events"]
    del tensors, refs, actual, expected, prof
    torch.cuda.empty_cache()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_memory_v0_model import model, batch, enable_attention_gates
    optimized = model(checkpointing=True).cuda().train()
    enable_attention_gates(optimized.action_expert)
    dense = copy.deepcopy(optimized)
    dense.predictor.attention_backend = dense.action_expert.attention_backend = "dense"
    b = {k: v.cuda() if torch.is_tensor(v) else v for k,v in batch().items()}
    b["future_valid"][1,24:] = False
    b["action_is_pad"][1,16:] = True
    noise, time = torch.randn_like(b["action"]), torch.tensor([.3,.7],device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        a = optimized(b,flow_noise=noise,flow_time=time)[0]
        c = dense(b,flow_noise=noise,flow_time=time)[0]
    a.backward(); c.backward()
    numerator, denominator = 0., 0.
    for p,r in zip(optimized.parameters(),dense.parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(r.grad).all()
            numerator += float((p.grad.float()-r.grad.float()).square().sum())
            denominator += float(r.grad.float().square().sum())
    report["model"].update(loss_abs_error=float((a-c).abs()),
                            gradient_relative_l2=(numerator/denominator)**.5,
                            gradient_relative_l2_tolerance=.03)
    assert report["model"]["gradient_relative_l2"] < .03
    optimized.eval(); dense.eval()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        errors=[]
        for x,y in zip(optimized._encode_context(b)[:2],dense._encode_context(b)[:2]):
            for s,t in zip(x.layers,y.layers):
                errors.extend([relative_error(s.key,t.key),relative_error(s.value,t.value)])
        actions = optimized.predict_action(b,initial_noise=noise)
        expected_actions = dense.predict_action(b,initial_noise=noise)
        report["model"].update(kv_max_relative_l2=max(errors),
                               action_relative_l2=relative_error(actions,expected_actions))
        assert max(errors)<.03 and report["model"]["action_relative_l2"]<.03
    report["status"]="passed"
    Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
