"""decode_overlap: decoding chunk k on one worker thread while chunk k+1 samples gives the same pool."""
import threading
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C, packed, sample
from orbitol.sampler import orbit_edm_sampler
from src.utils.ema import EMA
from test_decode import encoding, sampler_batch, tables  # noqa: F401
from test_sample_overrides import built_cfg, snapshot  # noqa: F401


def test_default_validation_and_override(snapshot):
    assert C.TrainConfig().decode_overlap == "none"
    assert C.TrainConfig(**C.normalize_historical_config({})).decode_overlap == "none"
    with pytest.raises(ValueError, match="decode_overlap"):
        C.TrainConfig(decode_overlap="process")
    assert "decode_overlap" in sample.SAMPLER_OVERRIDES
    assert built_cfg(snapshot, dict(decode_overlap="thread")).decode_overlap == "thread"


def pool(tables, encoding, monkeypatch, *, overlap, impl, budget, n=5, chunk=2, poison=3):
    """A real EDM loop with an oracle denoiser; attempt ``poison`` gets a non-finite lattice."""
    targets, skeleton = sampler_batch(tables, encoding, n=1)
    current = {}

    def oracle(state, sigma, skel, tb, *, objective, sigma_data):
        size = state["type_noisy"].shape[0]
        den = {"D_type": targets["type"].to(state["type_noisy"]).expand(size, -1, -1).clone(),
               "D_u": targets["u_c"].to(state["u_noisy"]).expand(size, -1, -1).clone(),
               "D_z": targets["z"].to(state["z_noisy"]).expand(size, -1).clone()}
        den["D_z"][current["index"] == poison, 0] = float("nan")
        return den

    def sampler(model, batch, *args, denoise_fn=None, **kwargs):
        current["index"] = batch["index"]
        return orbit_edm_sampler(model, batch, *args, denoise_fn=oracle, **kwargs)

    monkeypatch.setattr(packed, "PackedDenoiser", lambda *a, **kw: SimpleNamespace(denoise=None))
    model = torch.nn.Linear(1, 1)
    cfg = C.TrainConfig(K=2, sample_num_steps=2, S_churn=0., sample_token_budget=budget, compile_trunk="none",
                        decode_impl=impl, decode_overlap=overlap)
    bank = SimpleNamespace(sample=lambda count, **kwargs: [skeleton] * count)
    return sample.sample_pool(model, EMA(model, .9), cfg, tables, tables.torch("cpu"), encoding, None, bank,
                              n=n, seed=7, device="cpu", chunk=chunk, sampler=sampler)


def assert_identical(a, b):
    assert a.keys() == b.keys()
    for key in ("rows", "raw"):
        assert a[key].keys() == b[key].keys()
        for name in a[key]:
            assert torch.equal(a[key][name], b[key][name]), (key, name)
    for key in ("contains_a", "requested_G"):
        assert torch.equal(a[key], b[key]), key
    for key in ("skeletons", "failures", "failure_counts", "metadata"):
        assert a[key] == b[key], key


@pytest.mark.parametrize("impl", ["reference", "fast"])
@pytest.mark.parametrize("budget", [0, 8])
def test_thread_overlap_is_bitwise_identical(tables, encoding, monkeypatch, impl, budget):
    plain = pool(tables, encoding, monkeypatch, overlap="none", impl=impl, budget=budget)
    overlapped = pool(tables, encoding, monkeypatch, overlap="thread", impl=impl, budget=budget)
    assert_identical(plain, overlapped)
    # The injected failure keeps its attempt index across chunks.
    assert overlapped["failures"] == {3: "non_finite"}
    assert overlapped["raw"]["finite"].tolist() == [True, True, True, False, True]
    assert len(overlapped["skeletons"]) == overlapped["metadata"]["n"] == 5


def chunked(tables, encoding, monkeypatch, decoder, overlap, on_sample=lambda start: None, n=6, chunk=2):
    out, skeleton = sampler_batch(tables, encoding, n=n)
    starts = []

    def sampler(model, batch, *args, **kwargs):
        starts.append(int(batch["index"][0]))
        on_sample(starts[-1])
        return {k: v[batch["index"]] for k, v in out.items()}

    monkeypatch.setattr(sample, "decode_batch", decoder)
    model = torch.nn.Linear(1, 1)
    model.train()
    calls = []
    ema = SimpleNamespace(apply=lambda model: calls.append("apply"),
                          restore=lambda model, backup: calls.append("restore"))
    cfg = C.TrainConfig(sample_token_budget=0, compile_trunk="none", decode_overlap=overlap)
    bank = SimpleNamespace(sample=lambda count, **kwargs: [skeleton] * count)

    def run():
        return sample.sample_pool(model, ema, cfg, tables, tables.torch("cpu"), encoding, None, bank,
                                  n=n, seed=1, device="cpu", chunk=chunk, sampler=sampler)
    return run, starts, calls, model


@pytest.mark.parametrize("overlap, expected", [("none", False), ("thread", True)])
def test_thread_mode_decodes_while_the_next_chunk_samples(tables, encoding, monkeypatch, overlap, expected):
    """The first chunk's decode can see the second chunk's sampling start only when decoding overlaps."""
    real = sample.decode_batch
    second_chunk_started = threading.Event()
    observed, sizes = [], []

    def decoder(output, skels, *args, **kwargs):
        if not observed:
            observed.append(second_chunk_started.wait(timeout=10 if expected else 0.2))
        sizes.append(len(skels))
        return real(output, skels, *args, **kwargs)

    def on_sample(start):
        if start == 2:
            second_chunk_started.set()

    run, starts, calls, model = chunked(tables, encoding, monkeypatch, decoder, overlap, on_sample)
    result = run()
    assert observed == [expected]
    assert starts == [0, 2, 4] and sizes == [2, 2, 2]
    assert result["rows"]["A0"].shape[0] == 6
    assert calls == ["apply", "restore"] and model.training


def test_worker_errors_propagate_and_leave_no_thread(tables, encoding, monkeypatch):
    def decoder(*args, **kwargs):
        raise RuntimeError("decode exploded")

    before = threading.active_count()
    run, starts, calls, model = chunked(tables, encoding, monkeypatch, decoder, "thread")
    with pytest.raises(RuntimeError, match="decode exploded"):
        run()
    assert calls == ["apply", "restore"] and model.training
    assert threading.active_count() == before
    assert not any(t.name.startswith("orbitol-decode") for t in threading.enumerate())
