"""CPU-only integrity checks; downloads and compilation never run."""
import hashlib
import io
import shutil
import zipfile

import pytest
from tools import build_nequip as build


def test_weights_digest(tmp_path):
    members = [("data/0", b"weights"), ("extra/x", b"metadata"),
               ("code/a.py", b"code"), (".data/serialization_id", b"id")]
    def artifact(name, root, entries):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as z:
            for rel, data in entries:
                z.writestr(root + "/" + rel, data)
        return path
    first = artifact("first.zip", "root", members)
    changed_code = members[:2] + [("code/a.py", b"other"), (".data/serialization_id", b"other")]
    second = artifact("second.zip", "different", reversed(changed_code))
    expected = hashlib.sha256(b"data/0\0weightsextra/x\0metadata").hexdigest()
    assert build.weights_digest(first) == build.weights_digest(second) == expected
    for index in (0, 1):
        changed = list(members)
        changed[index] = (changed[index][0], b"changed")
        assert build.weights_digest(artifact(f"changed{index}.zip", "root", changed)) != expected


@pytest.mark.parametrize("bad", ["size", "md5", "sha256", None])
def test_source_checks_before_compile(tmp_path, monkeypatch, bad):
    payload = b"synthetic public source"
    source = dict(build.SOURCE, size=len(payload), md5=hashlib.md5(payload).hexdigest(),
                  sha256=hashlib.sha256(payload).hexdigest())
    if bad:
        source[bad] = 0 if bad == "size" else "0" * len(source[bad])
    monkeypatch.setattr(build, "SOURCE", source)
    monkeypatch.setattr(build.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(10**12, 0, 10**12))
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url, context: io.BytesIO(payload))
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: pytest.fail("compile reached"))
    if bad:
        with pytest.raises(ValueError, match="checksum mismatch"):
            build.main(["--source-dir", str(tmp_path), "--ts", str(tmp_path / "ts.nequip.pth"),
                        "--oeq", str(tmp_path / "oeq.nequip.pth"), "--no-check"])
        assert not (tmp_path / source["file"]).exists()
    else:
        path = build.download_source(tmp_path)
        monkeypatch.setattr(build.urllib.request, "urlopen", lambda *a, **k: pytest.fail("redownload"))
        assert build.download_source(tmp_path) == path
