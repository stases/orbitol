"""CPU contract tests; the CUDA arithmetic test runs only on a GPU."""
import copy
import io

import pytest
import torch

from orbitol.optim import ExactAdamW, load_adamw_state


def parameters():
    return [torch.nn.Parameter(torch.linspace(-1, 1, 35).reshape(5, 7)),
            torch.nn.Parameter(torch.tensor([0., 0.25, -0.5]))]


def gradients(params, iteration):
    for i, p in enumerate(params):
        p.grad = None if i == 1 and iteration % 3 == 0 else torch.sin(p.detach() + iteration)


def assert_state_equal(a, b):
    for ga, gb in zip(a.param_groups, b.param_groups):
        for pa, pb in zip(ga["params"], gb["params"]):
            assert torch.equal(pa.view(torch.int32), pb.view(torch.int32))
            assert a.state[pa].keys() == b.state[pb].keys()
            for key, value in a.state[pa].items():
                assert torch.equal(value.view(torch.int32), b.state[pb][key].view(torch.int32))


def test_cpu_foreach_with_changing_lr_and_missing_gradients():
    a, b = parameters(), parameters()
    exact = ExactAdamW(a)
    reference = torch.optim.AdamW(b, lr=1e-4, weight_decay=0., foreach=True)
    for iteration in range(120):
        gradients(a, iteration)
        gradients(b, iteration)
        exact.param_groups[0]["lr"] = reference.param_groups[0]["lr"] = 1e-4 / (iteration + 1)
        exact.step()
        reference.step()
    assert_state_equal(exact, reference)
    exact.zero_grad(set_to_none=True)
    previous = copy.deepcopy(exact.state_dict())
    exact.step()
    for key, state in previous["state"].items():
        for field, value in state.items():
            assert torch.equal(value, exact.state_dict()["state"][key][field])


@pytest.mark.parametrize("source_fused", [False, True])
@pytest.mark.parametrize("target", ["foreach", "fused", "exact"])
def test_cpu_checkpoint_flags_and_next_update(source_fused, target):
    source_params = parameters()
    source = torch.optim.AdamW(source_params, lr=2e-5, weight_decay=0.,
                               fused=source_fused, foreach=not source_fused)
    gradients(source_params, 1)
    source.step()
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    saved_flags = copy.deepcopy(saved["param_groups"])
    saved_steps = [s["step"].clone() for s in saved["state"].values()]
    a, b = ([torch.nn.Parameter(p.detach().clone()) for p in source_params] for _ in range(2))
    def make(params):
        if target == "exact":
            return ExactAdamW(params)
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.,
                                 fused=target == "fused", foreach=target != "fused")
    actual, direct = make(a), make(b)
    load_adamw_state(actual, saved)
    # A destination-configured ordinary load is the independent reference.
    expected = copy.deepcopy(saved)
    expected["param_groups"][0].update(fused=target == "fused", foreach=target != "fused",
                                       capturable=False)
    direct.load_state_dict(expected)
    assert saved["param_groups"] == saved_flags
    assert actual.param_groups[0]["lr"] == 2e-5
    assert actual.param_groups[0]["fused"] == (target == "fused")
    assert actual.param_groups[0]["foreach"] == (target != "fused")
    for p, old in zip(a, saved["state"].values()):
        assert actual.state[p]["step"].data_ptr() != old["step"].data_ptr()
        assert actual.state[p]["step"].device.type == "cpu"
    gradients(a, 2)
    gradients(b, 2)
    actual.step()
    direct.step()
    assert_state_equal(actual, direct)
    for before, state in zip(saved_steps, saved["state"].values()):
        assert torch.equal(before, state["step"])


def test_direct_exact_load_keeps_flags_and_rejects_unsupported_settings():
    source = torch.optim.AdamW(parameters(), weight_decay=0., fused=True)
    gradients(source.param_groups[0]["params"], 1)
    source.step()
    exact = ExactAdamW(parameters())
    exact.load_state_dict(source.state_dict())
    assert exact.param_groups[0]["fused"] is False
    assert exact.param_groups[0]["foreach"] is True
    assert exact.param_groups[0]["capturable"] is False
    bad = copy.deepcopy(source.state_dict())
    bad["param_groups"][0]["weight_decay"] = .1
    with pytest.raises(ValueError, match="settings"):
        exact.load_state_dict(bad)
    with pytest.raises(ValueError, match="zero weight decay"):
        ExactAdamW(parameters(), weight_decay=.01)
    with pytest.raises(ValueError, match="scalar learning rate"):
        ExactAdamW(parameters(), lr=torch.tensor(1e-4))
    with pytest.raises(ValueError, match="closures"):
        exact.step(lambda: None)


def test_uninitialized_state_can_resume_and_initialize_lazily():
    a, b = parameters(), parameters()
    source, target = ExactAdamW(a), ExactAdamW(b)
    # Inspecting an unused parameter's state can create an empty state entry.
    assert not source.state[a[1]]
    target.load_state_dict(source.state_dict())
    gradients(a, 1)
    gradients(b, 1)
    source.step()
    target.step()
    assert_state_equal(source, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton update runs on CUDA")
def test_cuda_update_is_bitwise_foreach():
    # Guards the pinned PyTorch/Triton pair: any rounding change shows up as differing bits.
    torch.manual_seed(0)
    shapes = [(1023,), (64, 33), (5, 7, 11), (1,)]
    start = [torch.randn(shape, device="cuda") for shape in shapes]
    reference = [p.clone().requires_grad_() for p in start]
    exact = [p.clone().requires_grad_() for p in start]
    optimizers = (torch.optim.AdamW(reference, lr=1e-4, weight_decay=0.0, foreach=True), ExactAdamW(exact, lr=1e-4))
    for step in range(40):
        gradients = [torch.randn(shape, device="cuda") * 10.0 ** (step % 5 - 2) for shape in shapes]
        for params, optimizer in zip((reference, exact), optimizers):
            optimizer.param_groups[0]["lr"] = 1e-4 * (step + 1) / 40
            for p, g in zip(params, gradients):
                p.grad = None if step % 7 == 3 and p.shape == (64, 33) else g.clone()
            optimizer.step()
    for a, b in zip(reference, exact):
        assert torch.equal(a.detach().view(torch.int32), b.detach().view(torch.int32))
        for key in ("exp_avg", "exp_avg_sq"):
            ra, rb = optimizers[0].state[a][key], optimizers[1].state[b][key]
            assert torch.equal(ra.view(torch.int32), rb.view(torch.int32))
