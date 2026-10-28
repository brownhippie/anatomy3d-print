"""The push guard, run here before a push: build the held job on this
working tree and read it against the commit before with the plugin
(anatomy3d.ledger.guard). The server does the same after every deploy
(the Lab builds the held job first); this is the same reading before the
code goes out.

    python tools/push_guard.py PHOTO --label NAME [--inputs inputs.pkl]
                               [--codes N] [--ledger DIR] [--workers 3]

Builds the first N settings of the photo's run order (refused ones left
out), labels them NAME in the ledger at DIR, and prints the guard of NAME
against the label built before it. Exit 1 when the guard fails. The first
label built is the first spins: build one on the commit you trust first.
"""
import argparse
import contextlib
import io
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _one(args):
    code, inputs_path, label, sha = args
    from anatomy3d import lab, ledger

    inputs = pickle.loads(Path(inputs_path).read_bytes())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            body = lab.build(code, inputs)
            m, _, _ = lab.score(code, body, inputs)
        rep = getattr(body, "report", None) or {}
        if "checks" in rep:
            m["checks_failed"] = rep["checks_failed"]
        row = ledger.make_row("lab", ledger.next_counter(), code, sha, m, held=True, status="ok")
    except Exception as exc:  # noqa: BLE001
        row = ledger.make_row("lab", ledger.next_counter(), code, sha, {}, held=True, status="fail", error=str(exc)[:200])
    row["commit"] = label
    ledger.append(row)
    return code, row.get("status")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--label", required=True)
    ap.add_argument("--inputs", help="a pickled lab.LabInputs for the photo (read from the photo when not given)")
    ap.add_argument("--codes", type=int, default=160)
    ap.add_argument("--ledger", default=str(ROOT / ".guard"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.environ["ANATOMY3D_LEDGER_DIR"] = a.ledger
    from anatomy3d import lab, ledger

    sha = ledger.photo_sha(a.photo)
    ip = Path(a.inputs) if a.inputs else Path(a.ledger) / f"inputs_{sha}.pkl"
    if not ip.exists():
        with contextlib.redirect_stdout(io.StringIO()):
            ip.write_bytes(pickle.dumps(lab.prepare_inputs(a.photo)))
    inputs = pickle.loads(ip.read_bytes())
    fams = (["O"] if inputs.capsule is None else ["C", "O"]) + (["B"] if inputs.reference is not None else [])
    order = [c for c in lab.run_order(fams, bust=bool(inputs.bust_hint)) if not lab.refused(c, inputs.animal_frac)]
    codes = order[:a.codes]
    h = ledger.held()
    if not h or h.get("photo") != sha:
        ledger.set_held(sha, codes)
    with ProcessPoolExecutor(a.workers) as ex:
        for code, st in ex.map(_one, [(c, str(ip), a.label, sha) for c in codes]):
            print(code, st, flush=True)
    g = ledger.guard(a.label)
    print("against", g.get("against"))
    print("PASSED" if g["passed"] else "FAILED")
    for w in g["why"]:
        print("  ", w)
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
