import hashlib

import orbitol.core as core
from orbitol import contracts


def test_core_matches_pin():
    core.verify_core()


def test_src_comes_from_the_pin():
    import src.crystalite.crystalite as cm
    assert cm.__file__.startswith(core.CORE_ROOT)


def test_type_encoding_asset_hash():
    import os
    import torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state = torch.load(os.path.join(root, contracts.TYPE_ENCODING_STATE), weights_only=False)
    assert state["state_sha256"] == contracts.TYPE_ENCODING_SHA256
