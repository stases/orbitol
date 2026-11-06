"""Empirical frequencies, typed alignment and integrity of the skeleton bank."""
from collections import Counter
import json

import numpy as np
import pytest

from orbitol.contracts import Skeleton
from orbitol.skeleton_prior import SkeletonBank, main


@pytest.fixture
def toy_cache(tmp_path):
    # The final two rows have the same untyped multiset in opposite token orders.
    groups = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    rows = np.array([[1, 0]] * 3 + [[1, 1]] + [[2, 3]] * 5 + [[3, 2]])
    atoms = np.array([[6, 0], [8, 0], [14, 0], [1, 8]] + [[11, 17]] * 5 + [[9, 3]])
    path = tmp_path / "train.npz"
    np.savez(path, G=groups, row=rows, A0=atoms, n_orbits=np.count_nonzero(rows, axis=1),
             round_trip_ok=np.ones(10, dtype=bool))
    return path


def test_counts_indices_and_multiset_key(toy_cache):
    bank = SkeletonBank.from_cache(toy_cache)
    assert bank.counts == {(1, (1,)): 3, (1, (1, 1)): 1, (2, (2, 3)): 6}
    assert bank.indices == {(1, (1,)): (0, 1, 2), (1, (1, 1)): (3,),
                            (2, (2, 3)): (4, 5, 6, 7, 8, 9)}
    assert bank.group_counts == {1: 4, 2: 6}


@pytest.mark.parametrize("tau,conditional", [(1, .75), (0, .5), (2, .9), (-1, .25)])
def test_empirical_groups_and_conditional_count_exponent(toy_cache, tau, conditional):
    bank = SkeletonBank.from_cache(toy_cache)
    samples = bank.sample(10000, generator=np.random.default_rng(82), temperature=tau)
    counts = Counter((s.G, s.rows) for s in samples)
    group_one = counts[(1, (1,))] + counts[(1, (1, 1))]
    assert group_one / len(samples) == pytest.approx(.4, abs=.02)
    assert counts[(1, (1,))] / group_one == pytest.approx(conditional, abs=.025)
    assert set(counts) == set(bank.counts) and all(s.A0 is None for s in samples)
    assert samples[:25] == bank.sample(10000, generator=np.random.default_rng(82), temperature=tau)[:25]


def test_uniform_training_assignments_and_unsorted_alignment(toy_cache):
    bank = SkeletonBank.from_cache(toy_cache)
    rng = np.random.default_rng(71)
    assignments = Counter(bank.typed_assignment(Skeleton(1, (1,)), generator=rng).A0 for _ in range(3000))
    assert set(assignments) == {(6,), (8,), (14,)}
    for count in assignments.values():
        assert count / 3000 == pytest.approx(1 / 3, abs=.035)
    assignments = Counter(bank.typed_assignment(Skeleton(2, (3, 2)), generator=rng).A0 for _ in range(3000))
    assert set(assignments) == {(17, 11), (9, 3)}
    assert assignments[(9, 3)] / 3000 == pytest.approx(1 / 6, abs=.025)
    assert bank.typed_assignment(Skeleton(1, (1, 1)), generator=rng).A0 == (1, 8)
    with pytest.raises(KeyError):
        bank.typed_assignment(Skeleton(1, (1, 1, 1)), generator=rng)


def test_save_load_hash_and_cli(toy_cache, tmp_path):
    bank = SkeletonBank.from_cache(toy_cache)
    path = tmp_path / "bank.json"
    main(["build", "--cache", str(tmp_path), "--out", str(path)])
    restored = SkeletonBank.load(path)
    assert restored.sha256 == bank.sha256
    assert restored.indices == bank.indices and restored.training == bank.training
    assert restored.sample(30, generator=np.random.default_rng(9)) == bank.sample(30, generator=np.random.default_rng(9))
    payload = json.loads(path.read_text())
    payload["training"][0]["A0"][0] = 7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        SkeletonBank.load(path)


def test_invalid_bank_inputs_and_split(toy_cache, tmp_path):
    bank = SkeletonBank.from_cache(toy_cache)
    for n, tau in ((-1, 1), (1.5, 1), (1, np.nan), (1, np.inf)):
        with pytest.raises(ValueError):
            bank.sample(n, generator=np.random.default_rng(0), temperature=tau)
    assert bank.sample(0, generator=np.random.default_rng(0)) == []
    with pytest.raises(ValueError, match="at least one"):
        SkeletonBank([])
    with np.load(toy_cache) as original:
        arrays = {k: original[k] for k in original.files}
    arrays["manifest"] = np.asarray(json.dumps(dict(split="val")))
    path = tmp_path / "val.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="train split"):
        SkeletonBank.from_cache(path)
