"""Real-Qwen preflight on existing external training data; not a benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .direct_composition_pilot import DirectRelationHyperLoRA
from .basis_state_controller_v101 import BasisStateController
from .joint_domains_v93 import MODEL, ROOT, filehash


def tensor_hash(values):
    h = hashlib.sha256()
    for v in values:
        h.update(v.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(device, output):
    if output.exists():
        raise FileExistsError(output)
    parent = ROOT / 'results/trajectory_hyperlora/external_hyper_v100_20261010'
    checkpoint = torch.load(parent / 'trained.pt', map_location='cpu', weights_only=True)
    cache = torch.load(parent / 'cache.pt', map_location='cpu', weights_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    agent = DirectRelationHyperLoRA(model, rank=4, layers=2,
        encoder_kind='contextual', task_conditioned=True).to(device)
    params = dict(agent.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    assert expected == set(checkpoint['hyper'])
    with torch.no_grad():
        for n, v in checkpoint['hyper'].items():
            params[n].copy_(v)
    agent.requires_grad_(False); agent.eval()
    initial_hash = tensor_hash(params[n] for n in sorted(expected))
    fields = lambda key: {k: v.to(device) for k, v in cache['fields'][key].items()}
    bases = []
    for row in cache['prepared'][:4]:
        with torch.no_grad():
            agent.set_source(fields(row['source']), target_fields=fields(row['query']))
            bases.append([a.b.detach().clone() for a in agent.adapters])
    norms = [[float(x.norm()) for x in row] for row in bases]
    assert all(sum(row) > 0 for row in norms), 'degenerate untrained bases'
    before_basis = tensor_hash(x for row in bases for x in row)
    c = BasisStateController(bases, lr=.1)
    row = cache['prepared'][0]
    ids = torch.tensor([row['prefix'] + row['answer']], device=device)
    labels = torch.tensor([[-100] * len(row['prefix']) + row['answer']], device=device)
    def loss_fn(bb):
        for a, b in zip(agent.adapters, bb, strict=True):
            # Keep direct-B Parameters outside the frozen model registry.
            a.b = b * 1.0
        return model(input_ids=ids, labels=labels, use_cache=False).loss
    initial = float(loss_fn(c.matrices()).detach())
    info = c.update_mounted(loss_fn)
    final = float(loss_fn(c.matrices()).detach())
    # Growth regression: compare real appended content against a zero-content
    # bank with identical coefficients/gate, isolating it from dilution.
    grown = BasisStateController(bases, lr=.1)
    empty = BasisStateController(bases, lr=.1)
    grown.append_basis(bases[0], initial_logit=.5)
    empty.append_basis([torch.zeros_like(b) for b in bases[0]], initial_logit=.5)
    weight = grown.state.coefficients.softmax(0)[-1] * grown.gate
    differences = [a - b for a, b in zip(grown.matrices(), empty.matrices(), strict=True)]
    for delta, source in zip(differences, bases[0], strict=True):
        torch.testing.assert_close(delta, weight * source, rtol=.03, atol=.002)
    content_delta_norm = sum(float(d.float().square().sum()) for d in differences) ** .5
    assert content_delta_norm > 0
    content_loss = float(loss_fn(grown.matrices()).detach())
    empty_content_loss = float(loss_fn(empty.matrices()).detach())
    # Genuine direct-B SGD check starts from precisely the same mixture.
    fresh = BasisStateController(bases, lr=.1)
    direct = [torch.nn.Parameter(x.detach().clone()) for x in fresh.matrices()]
    before_direct = tensor_hash(direct)
    opt = torch.optim.SGD(direct, lr=.1)
    opt.zero_grad(); direct_loss = loss_fn(direct); direct_loss.backward(); opt.step()
    direct_changed = before_direct != tensor_hash(direct)
    assert tensor_hash(params[n] for n in sorted(expected)) == initial_hash
    assert all(p.grad is None for p in agent.parameters())
    assert tensor_hash(x for row in bases for x in row) == before_basis
    result = dict(protocol='external training-row gradient preflight, no benchmark score',
        checkpoint_sha256=filehash(parent / 'trained.pt'),
        cache_sha256=filehash(parent / 'cache.pt'), basis_norms=norms,
        distinct_bases=len({tensor_hash(row) for row in bases}),
        initial_loss=initial, final_loss=final, coefficient_update=info,
        appended_content_delta_norm=content_delta_norm,
        appended_content_loss=content_loss, zero_content_loss=empty_content_loss,
        source_sha256=filehash(Path(__file__)),
        direct_B_changed=direct_changed, frozen_parameters_unchanged=True,
        bases_unchanged=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    assert info['matrix_changed'] and direct_changed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); main(args.device, args.output)
