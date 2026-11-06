"""Empirical untyped skeleton prior, with empirical typed assignments.

Rows are sorted only to form multiset keys; coordinate training targets are not
canonicalised. The temperature argument is the exponent tau: within each space
group, skeletons are drawn with weight count**tau.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np

from orbitol.contracts import Skeleton


class SkeletonBank:
    def __init__(self, training, *, table_sha256=None, source_sha256=None):
        self.training = tuple(training)
        if not self.training:
            raise ValueError("Skeleton bank requires at least one training structure")
        self.table_sha256, self.source_sha256 = table_sha256, source_sha256
        indices = defaultdict(list)
        for i, skeleton in enumerate(self.training):
            if (not 1 <= skeleton.G <= 230 or not skeleton.rows or skeleton.A0 is None
                    or len(skeleton.rows) != len(skeleton.A0)
                    or any(r <= 0 for r in skeleton.rows)
                    or any(not 1 <= z <= 118 for z in skeleton.A0)):
                raise ValueError(f"Invalid training skeleton at index {i}")
            indices[self._key(skeleton)].append(i)
        self.indices = {key: tuple(value) for key, value in sorted(indices.items())}
        self.counts = {key: len(value) for key, value in self.indices.items()}
        self.groups = np.asarray(sorted({key[0] for key in self.counts}), dtype=np.int64)
        self._keys = {int(g): [key for key in self.counts if key[0] == g] for g in self.groups}
        self.group_counts = {int(g): sum(self.counts[key] for key in self._keys[int(g)]) for g in self.groups}

    @staticmethod
    def _key(skeleton):
        return int(skeleton.G), tuple(sorted(int(r) for r in skeleton.rows))

    @classmethod
    def from_cache(cls, train_npz):
        train_npz = Path(train_npz)
        with np.load(train_npz, allow_pickle=False) as cache:
            manifest = json.loads(str(cache["manifest"].item())) if "manifest" in cache else {}
            if manifest.get("split", "train") != "train":
                raise ValueError("Skeleton prior must be built from the train split")
            if "round_trip_ok" in cache and not cache["round_trip_ok"].all():
                raise ValueError("Training cache contains failed round trips")
            G, row, A0, sizes = (cache[k] for k in ("G", "row", "A0", "n_orbits"))
            if (row.ndim != 2 or row.shape != A0.shape or len(G) != len(row)
                    or sizes.shape != G.shape or G.ndim != 1
                    or np.any(sizes <= 0) or np.any(sizes > row.shape[1])):
                raise ValueError("Invalid training cache shapes or orbit counts")
            training = []
            for g, rows, atoms, size in zip(G, row, A0, sizes):
                size = int(size)
                if np.any(rows[size:]) or np.any(atoms[size:]):
                    raise ValueError("Nonzero cache padding")
                training.append(Skeleton(int(g), tuple(map(int, rows[:size])), tuple(map(int, atoms[:size]))))
        digest = hashlib.sha256()
        with train_npz.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(training, table_sha256=manifest.get("table_sha256"), source_sha256=digest.hexdigest())

    def sample(self, n, *, generator: np.random.Generator, temperature=1.0) -> list[Skeleton]:
        if n < 0 or int(n) != n or not np.isfinite(temperature):
            raise ValueError("Invalid sample count or temperature exponent")
        p_group = np.asarray([self.group_counts[int(g)] for g in self.groups], dtype=np.float64)
        p_group /= p_group.sum()
        conditional = {}
        for g in self.groups:
            logp = np.log([self.counts[key] for key in self._keys[int(g)]])
            # Subtract before multiplication, preventing overflow at large finite tau.
            logp -= logp.max() if temperature >= 0 else logp.min()
            with np.errstate(over="ignore"):
                weights = np.exp(logp * temperature)
            conditional[int(g)] = weights / weights.sum()
        result = []
        for g in generator.choice(self.groups, size=int(n), p=p_group):
            keys = self._keys[int(g)]
            key = keys[int(generator.choice(len(keys), p=conditional[int(g)]))]
            result.append(Skeleton(key[0], key[1]))
        return result

    def typed_assignment(self, skeleton, *, generator: np.random.Generator) -> Skeleton:
        key = self._key(skeleton)
        if key not in self.indices:
            raise KeyError(f"Unknown empirical skeleton {key}")
        index = int(generator.choice(self.indices[key]))
        realised = self.training[index]
        by_row = defaultdict(deque)
        for row, z in zip(realised.rows, realised.A0):
            by_row[row].append(z)
        atoms = tuple(by_row[row].popleft() for row in skeleton.rows)
        return Skeleton(skeleton.G, skeleton.rows, atoms)

    def _payload(self):
        return dict(format_version=1, table_sha256=self.table_sha256, source_sha256=self.source_sha256,
                    training=[dict(G=s.G, rows=s.rows, A0=s.A0) for s in self.training])

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        payload["sha256"] = self.sha256
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported skeleton-bank format")
        training = [Skeleton(int(s["G"]), tuple(s["rows"]), tuple(s["A0"])) for s in payload["training"]]
        bank = cls(training, table_sha256=payload["table_sha256"], source_sha256=payload["source_sha256"])
        if bank.sha256 != payload.get("sha256"):
            raise ValueError("Skeleton-bank sha256 mismatch")
        return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cache = args.cache / "train.npz" if args.cache.is_dir() else args.cache
    bank = SkeletonBank.from_cache(cache)
    bank.save(args.out)
    print(json.dumps(dict(skeletons=len(bank.counts), structures=len(bank.training),
                          sha256=bank.sha256, path=str(args.out.resolve()))))


if __name__ == "__main__":
    main()
