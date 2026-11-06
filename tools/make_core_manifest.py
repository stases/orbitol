"""Write orbitol/core_manifest.json: sha256 of every file of the pinned Crystalite src tree."""
import argparse
import hashlib, json, os

from orbitol import contracts as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?")
    args = parser.parse_args()
    root = args.root or C.local_path("ORBITOL_CORE_ROOT")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orbitol", "core_manifest.json")
    files = {}
    for dp, _, fn in os.walk(os.path.join(root, "src")):
        for f in sorted(fn):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(dp, f)
            files[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump({"crystalite_commit": "03603fe", "files": dict(sorted(files.items()))}, open(out, "w"), indent=1)
    print(f"{len(files)} files -> {out}")


if __name__ == "__main__":
    main()
