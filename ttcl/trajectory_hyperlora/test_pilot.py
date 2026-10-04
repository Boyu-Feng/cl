"""CPU checks of the generated low-rank update path."""

import torch
from torch import nn

from ttcl.trajectory_hyperlora.pilot import ConditionalLoRALinear


def test_disabled_adapter_matches_frozen_linear() -> None:
    base = nn.Linear(3, 2, bias=False)
    layer = ConditionalLoRALinear(base, experts=2, rank=1)
    x = torch.randn(1, 4, 3)
    assert torch.equal(layer(x), base(x))


def test_source_coefficients_change_output_and_receive_gradients() -> None:
    base = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    layer = ConditionalLoRALinear(base, experts=2, rank=1)
    with torch.no_grad():
        layer.a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.b.copy_(torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]]))
    x = torch.tensor([[[2.0, 3.0]]])
    coefficients = torch.tensor([[0.25, 0.75]], requires_grad=True)
    layer.coefficients = coefficients
    result = layer(x)
    assert torch.allclose(result, torch.tensor([[[0.5, 2.25]]]))
    result.sum().backward()
    assert coefficients.grad is not None and torch.count_nonzero(coefficients.grad) == 2


def test_exported_rank_concatenation_matches_conditional_update() -> None:
    base = nn.Linear(3, 2, bias=False)
    layer = ConditionalLoRALinear(base, experts=3, rank=2)
    with torch.no_grad():
        layer.a.normal_()
        layer.b.normal_()
    coefficients = torch.tensor([[0.2, 0.3, 0.5]])
    layer.coefficients = coefficients
    x = torch.randn(1, 5, 3)
    a_export = layer.a.detach().reshape(6, 3)
    b_export = (layer.b.detach() * coefficients[0, :, None, None] / 2)
    b_export = b_export.permute(1, 0, 2).reshape(2, 6)
    exported_update = torch.nn.functional.linear(
        torch.nn.functional.linear(x, a_export), b_export
    )
    assert torch.allclose(layer(x) - base(x), exported_update, atol=1e-6)
