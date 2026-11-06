"""Independent sampler contracts, noise assignment, and precision controls."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import contracts as C
from orbitol.model import build_model
from orbitol.packed import PackedDenoiser, pack_layout, token_chunks, _bias_for_scores
from orbitol.sample import sample_pool, skeleton_batch
from src.utils.ema import EMA


@pytest.fixture(scope="module")
def tables(frozen_tables):
    return frozen_tables


def fixture(tables, kind="A", lengths=(1,3,7), device="cpu"):
    torch.manual_seed(4207)
    tb = tables.torch(device)
    cfg = C.TrainConfig(d_model=64, n_heads=4, n_layers=2, sample_num_steps=3,
                        gem_mode="none" if kind == "B" else "distance",
                        coord_input="internal" if kind == "C" else "explicit",
                        coord_readout="direct" if kind == "C" else "projection",
                        objective="posterior_periodic", precision="fp32")
    model = build_model(cfg,tb.row_desc,tb.group_desc,tables=tb).to(device).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.05*torch.randn_like(p))
    skeletons = [C.Skeleton(G=1, rows=(1,)*n) for n in lengths]
    batch = skeleton_batch(skeletons,tb,K=20,device=device)
    generator = torch.Generator(device=device).manual_seed(36)
    inputs = dict(type_in=torch.randn(len(lengths),20,16,device=device,generator=generator),
                  x_in=torch.rand(len(lengths),20,3,device=device,generator=generator),
                  lat_in=torch.randn(len(lengths),6,device=device,generator=generator)*.1,
                  pad_mask=batch["pad_mask"], c_noise=torch.zeros(len(lengths),device=device),
                  row=batch["row"], G=batch["G"],
                  u=torch.rand(len(lengths),20,3,device=device,generator=generator)-.5,
                  z=torch.randn(len(lengths),6,device=device,generator=generator)*.1)
    return model,cfg,tb,skeletons,batch,inputs


class FixedBank:
    def __init__(self, skeletons):
        self.skeletons = skeletons

    def sample(self,n,*,generator):
        assert n == len(self.skeletons)
        return self.skeletons


def fake_decode(output, selected, tables, encoding, allowed_mask, *, n_max):
    # Keep the chunk's order visible, and fail its last row to test global indices.
    count = len(selected)
    return SimpleNamespace(rows={"id": output["z"][:,0].clone()},
                           contains_a=np.ones(count,dtype=bool),
                           requested_G=np.asarray([s.G for s in selected]),
                           failures={count-1:"deliberate_failure"})


@pytest.mark.parametrize("kind",["A","B","C"])
@torch.no_grad()
def test_independent_nonzero_gates_filler_and_padded(kind,tables):
    model,_,_,_,batch,inputs = fixture(tables,kind)
    reference = model(**inputs)
    minimum = int((~batch["pad_mask"]).sum())+len(batch["G"])
    outputs = []
    for capacity in (minimum,65,256):
        result = PackedDenoiser(model,batch,token_budget=capacity)(**inputs)
        assert all(not t.requires_grad for t in result.values())
        for key in reference:
            torch.testing.assert_close(result[key],reference[key],atol=3e-6,rtol=3e-5)
        outputs.append(result)
    for key in reference:
        torch.testing.assert_close(outputs[0][key],outputs[-1][key],atol=3e-6,rtol=3e-5)


def test_exhaustive_small_chunk_boundaries():
    import itertools
    for lengths in itertools.product(range(1,5),repeat=4):
        skeletons = [C.Skeleton(G=1,rows=(1,)*n) for n in lengths]
        for budget in range(max(lengths)+1,12):
            chunks = token_chunks(skeletons,budget)
            assert chunks[0][0] == 0 and chunks[-1][1] == len(skeletons)
            assert all(b == c for (_,b),(c,_) in zip(chunks,chunks[1:]))
            for a,b in chunks:
                used = sum(n+1 for n in lengths[a:b])
                assert used <= budget
                if b < len(lengths):
                    assert used+lengths[b]+1 > budget
                assert b-a <= budget//2
    assert token_chunks([],7) == []
    with pytest.raises(ValueError,match="at least one"):
        token_chunks([C.Skeleton(G=1,rows=())],4)
    with pytest.raises(ValueError,match="structure 0"):
        token_chunks([C.Skeleton(G=1,rows=(1,)*4)],4)


def test_filler_isolation_and_structure_capacity():
    pad = torch.tensor([[False,True,False],[True,False,True]])
    for capacity in (5,6,19):
        layout = pack_layout(pad,capacity)
        assert layout.structure_capacity == capacity//2
        assert layout.column[:5].tolist() == [0,2,3,1,3]
        assert layout.document[:5].tolist() == [0,0,0,1,1]
        for i in range(5,capacity):
            assert int((layout.document == layout.document[i]).sum()) == 1
            assert layout.document[i] >= len(pad)


@pytest.mark.parametrize("budget,chunk,expected_failures",[(0,2,[1,3,4]),(7,100,[1,2,4])])
def test_public_pool_order_and_failure_indices(monkeypatch,tables,budget,chunk,expected_failures):
    import orbitol.sample as sampling
    model,cfg,tb,_,_,_ = fixture(tables)
    cfg = replace(cfg,sample_token_budget=budget)
    lengths = (1,3,4,2,1)
    skeletons = [C.Skeleton(G=i+1,rows=(1,)*n) for i,n in enumerate(lengths)]
    seen = []

    def sampler(model,batch,tb,**kwargs):
        seen.append((batch["G"].tolist(),"denoise_fn" in kwargs))
        n = len(batch["G"])
        return {"type":batch["G"][:,None,None].expand(n,20,16).float(),
                "u":torch.zeros(n,20,3), "z":batch["G"][:,None].expand(n,6).float(),
                "finite":torch.ones(n,dtype=torch.bool)}

    monkeypatch.setattr(sampling,"decode_batch",fake_decode)
    ema = EMA(model,.9)
    result = sample_pool(model,ema,cfg,tables,tb,None,None,FixedBank(skeletons),
                         n=5,seed=22,device="cpu",chunk=chunk,sampler=sampler)
    assert result["rows"]["id"].tolist() == [1,2,3,4,5]
    assert result["requested_G"].tolist() == [1,2,3,4,5]
    assert result["raw"]["z"][:,0].tolist() == [1,2,3,4,5]
    assert result["skeletons"] == [asdict(s) for s in skeletons]
    assert list(result["failures"]) == expected_failures
    assert result["failure_counts"] == {"deliberate_failure":len(expected_failures)}
    assert all(packed == bool(budget) for _,packed in seen)


@pytest.mark.parametrize("kind",["A","B","C"])
def test_real_sampler_draws_equal_only_when_chunks_match(monkeypatch,tables,kind):
    import orbitol.sample as sampling
    model,cfg,tb,skeletons,_,_ = fixture(tables,kind,lengths=(1,1,1,1))
    monkeypatch.setattr(sampling,"decode_batch",fake_decode)
    original_randn = torch.randn
    draws = []

    def record(*args,**kwargs):
        output = original_randn(*args,**kwargs)
        draws.append(output.clone())
        return output

    monkeypatch.setattr(torch,"randn",record)
    runs = []
    ema = EMA(model,.9)
    for budget in (0,8,4):
        draws.clear()
        result = sample_pool(model,ema,replace(cfg,sample_token_budget=budget),tables,tb,
            None,None,FixedBank(skeletons),n=4,seed=919,device="cpu",chunk=4)
        runs.append((result,list(draws)))
    assert len(runs[0][1]) == len(runs[1][1]) == 12  # Initial + 3 churn triples.
    assert all(torch.equal(a,b) for a,b in zip(runs[0][1],runs[1][1]))
    # Noise assignment is an exact contract. Trajectory drift is not bounded here by
    # an arbitrary tolerance on this untrained three-step trajectory.
    assert all(result["raw"]["finite"].all() for result,_ in runs)
    assert len(runs[2][1]) == 24
    # Different initial coordinate noise even for the first two sample indices.
    assert not torch.equal(runs[0][1][1][:2],runs[2][1][1])
    assert not torch.equal(runs[0][0]["raw"]["type"],runs[2][0]["raw"]["type"])


def test_exception_restores_ema_model_mode_and_rng(monkeypatch,tables):
    import orbitol.sample as sampling
    model,cfg,tb,skeletons,_,_ = fixture(tables,lengths=(1,))
    ema = EMA(model,.8)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    original = {k:v.clone() for k,v in model.state_dict().items()}
    model.train()
    state = torch.get_rng_state().clone()

    def fail(*args,**kwargs):
        torch.rand(2)
        raise RuntimeError("decode error")

    monkeypatch.setattr(sampling,"decode_batch",fail)
    with pytest.raises(RuntimeError,match="decode error"):
        sample_pool(model,ema,replace(cfg,sample_token_budget=2),tables,tb,None,None,
                    FixedBank(skeletons),n=1,seed=15,device="cpu")
    assert model.training
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())


def test_p0_matches_sdpa_autocast_dispatch():
    torch.manual_seed(31)
    q,k,v = (torch.randn(2,4,9,16) for _ in range(3))
    bias = torch.randn(2,4,9,9)*.7
    with torch.autocast("cpu",dtype=torch.bfloat16):
        actual = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias)
    quantized = _bias_for_scores(bias,torch.bfloat16)
    assert quantized.dtype == torch.bfloat16
    assert torch.equal(quantized,bias.to(torch.bfloat16))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.bfloat16(),k.bfloat16(),v.bfloat16(),attn_mask=quantized)
    assert torch.equal(actual,expected)
    assert not torch.equal(bias,bias.bfloat16().float())


@pytest.mark.data
def test_resume_treats_budget_as_execution_choice(tmp_path):
    from orbitol.train import Trainer
    cfg = C.TrainConfig(sample_count=0,num_workers=0)
    snapshot = tmp_path/"resume.pt"
    torch.save({"cfg":asdict(cfg)},snapshot)
    with pytest.raises(ValueError,match="Resume mismatch: cfg"):
        Trainer(replace(cfg,lr=2*cfg.lr,sample_token_budget=8192),run_name="reject",
                out_root=tmp_path,resume=str(snapshot),device="cpu",no_wandb=True)
    # A changed budget alone passes the cfg check; this minimal snapshot then fails on its missing state.
    with pytest.raises(Exception) as failure:
        Trainer(replace(cfg,sample_token_budget=8192),run_name="accept",
                out_root=tmp_path,resume=str(snapshot),device="cpu",no_wandb=True)
    assert "Resume mismatch: cfg" not in str(failure.value)


def test_training_novelty_excludes_failed_conversion_from_un_denominator():
    from orbitol.eval_adapter import _sample_items, _with_core_fallback
    from src.eval.dng_eval import compute_novelty_metrics
    from pymatgen.core import Lattice, Structure
    rows = dict(A0=torch.tensor([[1],[0]]), F1=torch.zeros(2,1,3), Y1=torch.zeros(2,6),
                pad_mask=torch.tensor([[False],[True]]))
    reference = Structure(Lattice.cubic(3),["Li"],[[0,0,0]])
    result = compute_novelty_metrics(_with_core_fallback(_sample_items(rows)),[reference],
                                    limit=2,minimum_nary=1)
    assert result.eval_count == 2 and len(result.novelty_structs) == 1
    assert result.un_rate == 1.0
    assert result.novelty_metrics["counts"]["novel_total"] == 1


@pytest.mark.gpu
@pytest.mark.parametrize("kind",["A","B","C"])
@torch.no_grad()
def test_cuda_short_and_long_compiled_replay(kind,tables):
    from orbitol.packed import sampling_trunk
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.compiler.reset()
    sampling_trunk.cache_clear()
    model,_,_,_,batch,inputs = fixture(tables,kind,device="cuda")
    for dtype in (None,torch.bfloat16):
        for capacity in (14,64,128,1024,64):
            adapter = PackedDenoiser(model,batch,token_budget=capacity,compile_mode="reduce-overhead")
            eager = PackedDenoiser(model,batch,token_budget=capacity)
            with torch.autocast("cuda",enabled=dtype is not None,dtype=dtype or torch.bfloat16):
                reference = {k:v.clone() for k,v in eager(**inputs).items()}
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    out = {k:v.clone() for k,v in adapter(**inputs).items()}
                    for k in reference:
                        assert torch.isfinite(out[k]).all()
                        torch.testing.assert_close(out[k],reference[k],atol=.015 if dtype else 3e-6,
                                                   rtol=.04 if dtype else 3e-5)
                torch.compiler.cudagraph_mark_step_begin()
                retained = adapter(**inputs)
                saved = {k:v.clone() for k,v in retained.items()}
                torch.compiler.cudagraph_mark_step_begin()
                adapter(**{**inputs,"type_in":inputs["type_in"]+.25})
                for k in saved:
                    assert torch.equal(retained[k],saved[k]), "returned buffer was overwritten"
