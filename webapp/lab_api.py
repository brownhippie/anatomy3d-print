"""Combination Lab: automated runs of every build combination for one photo.

Upload a photo -> a run is created with the full ordered list of
combinations (anatomy3d.lab.run_order). One background worker builds and
scores them one at a time, taking the same lock as normal generation per
combination so a person's own Generate is never stuck behind a long run.
Everything a run produces lives under LAB_ROOT/<run_id>/ so it survives
restarts and redeploys when LAB_ROOT is on a persistent volume:
  meta.json      photo name, families, status, created
  order.json     every combination code, in run order
  results.jsonl  one line per finished combination (metrics or error)
  ratings.json   your stars/notes per code
  m/<code>.obj.gz, t/<code>.png   preview model and thumbnail
"""
import contextlib
import gzip
import io
import json
import os
import re
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from trimesh.exchange.obj import export_obj

from anatomy3d import lab, ledger

import jobs
import study_api


def _lab_root() -> Path:
    env = os.environ.get("ANATOMY3D_LAB_DIR")
    if env:
        root = Path(env)
    elif Path("/data").is_dir() and os.access("/data", os.W_OK):
        root = Path("/data/lab")
    else:
        root = Path(tempfile.gettempdir()) / "anatomy3d-lab"
    root.mkdir(parents=True, exist_ok=True)
    return root


LAB_ROOT = _lab_root()
# Stop storing preview models (metrics and thumbnails still kept) once the
# lab's own folder passes this, so a long run can't fill the volume.
LAB_MAX_BYTES = int(float(os.environ.get("ANATOMY3D_LAB_MAX_GB", "9")) * 1e9)
# Below this much free space on the volume a run keeps only its metrics (no
# thumbnails or models), so a full volume never stops it or anything else.
LAB_MIN_FREE = 150e6
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
CODE_RE = re.compile(r"^[A-Z]-[0-9]{1,16}$")

router = APIRouter(prefix="/api/lab")
_state_lock = threading.Lock()
_generation_lock: Optional[threading.Lock] = None
_inputs_cache: dict = {}  # run_id -> LabInputs (only the active run's)
_lab_bytes = {"total": None}


# ---------------------------------------------------------------- storage


def _run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(404, "No such run.")
    d = LAB_ROOT / run_id
    if not (d / "meta.json").exists():
        raise HTTPException(404, "No such run.")
    return d


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _results(d: Path) -> "dict[str, dict]":
    out = {}
    p = d / "results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["code"]] = r
    return out


def _disk_free() -> float:
    try:
        return shutil.disk_usage(LAB_ROOT).free
    except OSError:
        return 0.0


def _folder_bytes() -> int:
    if _lab_bytes["total"] is None:
        _lab_bytes["total"] = sum(f.stat().st_size for f in LAB_ROOT.rglob("*") if f.is_file())
    return _lab_bytes["total"]


# ---------------------------------------------------------------- worker


def _active_run() -> Optional[str]:
    best, best_t = None, -1.0
    for d in LAB_ROOT.iterdir():
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta and meta.get("status") == "running" and meta.get("started", 0) > best_t:
            best, best_t = d.name, meta["started"]
    return best


_INPUTS_READING = "r5"


def _inputs(run_id: str, d: Path):
    if run_id not in _inputs_cache:
        _inputs_cache.clear()
        # read from the photo once and kept with the run: each build runs in
        # a process of its own (lab_job.py) and loads it instead of reading
        # the photo again (a minute and the photo-reading models each time)
        # (named for what reads it: when the reading changes -- the tie read
        # whole, 20 Oct -- a run reads its photo again once)
        p = d / f"inputs_{_INPUTS_READING}.pkl"
        inputs = None
        if p.exists():
            try:
                inputs = pickle.loads(p.read_bytes())
            except Exception:  # noqa: BLE001 - (an older code's inputs: read again)
                inputs = None
        if inputs is None:
            meta = _read_json(d / "meta.json", {})
            inputs = lab.prepare_inputs(str(d / meta["photo_file"]))
            try:
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(inputs))
                tmp.replace(p)
            except OSError:
                pass
        _inputs_cache[run_id] = inputs
    return _inputs_cache[run_id]


def _work_one(run_id: str) -> bool:
    d = LAB_ROOT / run_id
    meta = _read_json(d / "meta.json", {})
    order = _read_json(d / "order.json", [])
    done = _results(d)
    pending = [c for c in order if c not in done and c.split("-")[0] in meta.get("families", [])]
    if not pending:
        with _state_lock:
            meta = _read_json(d / "meta.json", {})
            meta["status"] = "done"
            meta["finished"] = time.time()
            _write_json(d / "meta.json", meta)
        if meta.get("calibration"):
            _calibration_done(d, meta)
        return False
    code = pending[0]
    t0 = time.time()
    row = {"code": code, "n": lab.number(code), "at": t0}
    key = f"run:{run_id}:{code}"
    why = jobs.stopped(key)
    if why:
        # (it took its process down before: failed, not built again)
        row.update(status="fail", error=f"stopped: {why}", seconds=0.0)
        with _state_lock:
            with open(d / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        return True
    jobs.mark(key)
    with _generation_lock:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if "families_checked" not in meta:
                    inputs = _inputs(run_id, d)
                    with _state_lock:
                        meta = _read_json(d / "meta.json", {})
                        fams = (["O"] if inputs.capsule is None else ["C", "O"]) + (["B"] if inputs.reference is not None else [])
                        meta.update(families=fams, families_checked=True, capsule_unavailable=inputs.capsule_unavailable)
                        if inputs.bust_hint and not meta.get("reordered_for_bust"):
                            _write_json(d / "order.json", lab.run_order(fams, bust=True))
                            meta["reordered_for_bust"] = True
                        # settings this photo never builds (layer rescue on a person): left out, not built to fail
                        now = _read_json(d / "order.json", [])
                        kept = [c for c in now if not lab.refused(c, inputs.animal_frac)]
                        if len(kept) != len(now):
                            _write_json(d / "order.json", kept)
                            meta["refused_left_out"] = len(now) - len(kept)
                        _write_json(d / "meta.json", meta)
                    return True
                inputs = _inputs(run_id, d)
                body = lab.build(code, inputs)
                metrics, preview, thumb = lab.score(code, body, inputs)
            row.update(status="ok", **metrics)
            rep = getattr(body, "report", None) or {}
            if "checks" in rep:
                # each step's reality check: how many ran, which failed
                row["checks_n"] = len(rep["checks"])
                row["checks_failed"] = rep["checks_failed"]
            added = 0
            if _disk_free() > LAB_MIN_FREE:
                (d / "t").mkdir(exist_ok=True)
                thumb.save(d / "t" / f"{code}.png", optimize=True)
                added = (d / "t" / f"{code}.png").stat().st_size
            if _folder_bytes() < LAB_MAX_BYTES and _disk_free() > LAB_MIN_FREE:
                (d / "m").mkdir(exist_ok=True)
                data = gzip.compress(export_obj(preview, include_normals=False, include_texture=False, digits=3).encode(), 6)
                (d / "m" / f"{code}.obj.gz").write_bytes(data)
                added += len(data)
                row["model"] = True
            else:
                row["model"] = False
            _lab_bytes["total"] = _folder_bytes() + added
        except Exception as exc:  # noqa: BLE001 - one bad combination mustn't stop the run
            row.update(status="fail", error=f"{type(exc).__name__}: {str(exc)[:200]}")
    row["seconds"] = round(time.time() - t0, 1)
    row["commit"] = ledger.machine()
    with _state_lock:
        with open(d / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    _ledger_row(d, meta, row)
    jobs.clear()
    return True


def _ledger_row(d: Path, meta: dict, row: dict) -> None:
    """The build as one change in the ledger (failed builds too: their
    error is the noise channel)."""
    try:
        sha = meta.get("photo_sha") or ledger.photo_sha(d / meta["photo_file"])
        h = ledger.held()
        r = ledger.make_row("lab", ledger.next_counter(), row["code"], sha, row if row.get("status") == "ok" else {},
                            run_id=d.name, status=row.get("status"), error=row.get("error"),
                            held=bool(h) and h.get("photo") == sha and row["code"] in set(h.get("codes") or []))
        ledger.append(r)
    except Exception as exc:  # noqa: BLE001 - the reading never stops a build
        print(f"Ledger: {row.get('code')}: {exc}")


def _release_memory() -> None:
    """Give freed memory back to the system after each job: a build peaks near
    2 GB in large numpy arrays, and without this the process kept what it had
    freed until the service ran out (8 GB) and was restarted mid-run."""
    import ctypes
    import gc

    # the models only reading a photo needs (cut-out, depth, the parts'
    # reader) and their working memory: 1.2 GB held between builds that never
    # use them -- loaded again when the next photo is read
    try:
        from anatomy3d import depth_source, head_parts, person_segmenter

        depth_source._session = None
        person_segmenter._birefnet_session = None
        head_parts._clipseg_sess = None
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


JOB_TIMEOUT = 40 * 60  # a build stopped after this long (the longest take a few minutes)


def _run_job(kind: str, arg: str) -> None:
    """One build, in a process of its own (lab_job.py), the generation lock
    held while it runs as it was for a build here. Whatever it takes is given
    back when it ends; if it is killed (out of memory) or stuck, only it
    goes, and the build it was on is stopped for good (jobs.py) -- recorded
    as failed the next time round, not tried again."""
    with _generation_lock:
        try:
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve().with_name("lab_job.py")), kind, arg],
                                timeout=JOB_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    if rc != 0:
        why = "it ran out of memory" if rc in (-9, 137) else ("it took over 40 minutes" if rc == "timeout" else f"its process ended ({rc})")
        key = jobs.current()
        if key:
            jobs.stop(key, why)
        print(f"Lab worker: {kind} {arg}: {key or 'no build'} stopped, {why}.")
    jobs.clear()
    _lab_bytes["total"] = None  # (what it stored, counted again)


# ---------------------------------------------------------------- the ledger: held job, calibration, app renders


def _run_dirs() -> "list[tuple[Path, dict]]":
    out = []
    for d in LAB_ROOT.iterdir():
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta:
            out.append((d, meta))
    return sorted(out, key=lambda dm: dm[1].get("created", 0))


def _sha(d: Path, meta: dict) -> Optional[str]:
    if meta.get("photo_sha"):
        return meta["photo_sha"]
    try:
        return ledger.photo_sha(d / meta["photo_file"])
    except (OSError, KeyError):
        return None


def _held_setup() -> None:
    """Once: the Lab's earlier builds into the ledger, and the held job (the
    photo with the most builds on the volume, its first HELD_N settings)."""
    if not (ledger.root() / "backfilled").exists():
        runs = []
        for d, meta in _run_dirs():
            sha = _sha(d, meta)
            if sha:
                runs.append((dict(meta, run_id=d.name), list(_results(d).values()), sha))
        n = ledger.backfill(runs)
        (ledger.root() / "backfilled").write_text(str(n))
        print(f"Ledger: {n} earlier builds read in.")
    if not ledger.held():
        best = max(_run_dirs(), key=lambda dm: len(_results(dm[0])), default=None)
        if best is None:
            return
        # the photo with the most builds; its settings from its latest run (every family it has now)
        sha = _sha(*best)
        d, meta = [dm for dm in _run_dirs() if _sha(*dm) == sha and not dm[1].get("calibration")][-1]
        fams = meta.get("families", ["O"])
        animal = 0.0 if "B" in fams else 1.0  # (a body was posed on it: a person)
        try:
            animal = pickle.loads((d / f"inputs_{_INPUTS_READING}.pkl").read_bytes()).animal_frac
        except Exception:  # noqa: BLE001
            pass
        order = [c for c in _read_json(d / "order.json", []) if c.split("-")[0] in fams and not lab.refused(c, animal)]
        ledger.set_held(_sha(d, meta), order[:ledger.HELD_N])
        print(f"Ledger: held job set from run {d.name}.")


def _ensure_calibration() -> None:
    """Every new commit builds the held job first: a run of the held photo
    with only the held settings, ahead of whatever run was going (it goes on
    after). Its window is what the push guard reads."""
    _held_setup()
    h = ledger.held()
    c = ledger.machine()
    if not h or c.startswith("unknown"):
        return
    src = None
    for d, meta in _run_dirs():
        if meta.get("calibration") and meta.get("commit") == c:
            return
        if _sha(d, meta) == h["photo"]:
            src = (d, meta)
    if src is None:
        return
    sd, smeta = src
    run_id = uuid.uuid4().hex[:12]
    d = LAB_ROOT / run_id
    d.mkdir(parents=True)
    shutil.copy(sd / smeta["photo_file"], d / smeta["photo_file"])
    for p in sd.glob("inputs_*.pkl"):
        shutil.copy(p, d / p.name)
    now = time.time()
    with _state_lock:
        resume = None
        for od, om in _run_dirs():
            if om.get("status") == "running":
                resume = od.name
                om["status"] = "paused"
                _write_json(od / "meta.json", om)
        _write_json(d / "order.json", list(h["codes"]))
        _write_json(d / "meta.json", {
            "run_id": run_id, "photo_name": f"held job · {c}", "photo_file": smeta["photo_file"], "created": now, "started": now,
            "status": "running", "families": smeta.get("families", ["O"]), "families_checked": True,
            "calibration": True, "commit": c, "photo_sha": h["photo"], "resume": resume,
        })
    print(f"Ledger: held job for {c} started (run {run_id}); then run {resume} goes on.")


def _calibration_done(d: Path, meta: dict) -> None:
    try:
        g = ledger.guard(meta["commit"])
        h = ledger.held()
        current = next((x for x in h.get("codes", []) if x.startswith("O-")), lab.CODE_PRODUCTION)
        p = ledger.choose(meta["commit"], current)
        print(f"Ledger: guard for {meta['commit']}: {'passed' if g['passed'] else 'failed'}; pick {p.get('code')} ({p.get('why')}).")
    except Exception as exc:  # noqa: BLE001
        print(f"Ledger: guard error: {exc}")
    r = meta.get("resume")
    if r and (LAB_ROOT / r / "meta.json").exists():
        with _state_lock:
            m = _read_json(LAB_ROOT / r / "meta.json", {})
            if m.get("status") == "paused":
                m["status"] = "running"
                m["started"] = time.time()
                _write_json(LAB_ROOT / r / "meta.json", m)


def _pending_measures() -> "list[str]":
    p = ledger.root() / "pending"
    if not p.is_dir():
        return []
    return sorted((x.name for x in p.iterdir() if (x / "meta.json").exists()), key=lambda n: (p / n / "meta.json").stat().st_mtime)


def measure_one(job_id: str) -> None:
    """An app render, measured as the Lab measures its builds, into the
    ledger. Runs in the worker's own process, after the model was handed
    over (the reading costs the render nothing)."""
    import trimesh

    from anatomy3d.mesh_types import BodyMesh

    p = ledger.root() / "pending" / job_id
    meta = _read_json(p / "meta.json", None)
    if meta is None:
        return
    key = f"measure:{job_id}"
    why = jobs.stopped(key)
    if why:
        # (it took its process down before: recorded, not tried again)
        ledger.append({"source": "app", "counter": meta.get("counter"), "job_id": job_id, "at": meta.get("at"), "commit": ledger.machine(),
                       "code": meta.get("code"), "status": "unmeasured", "error": f"stopped: {why}", "R": {}})
        shutil.rmtree(p, ignore_errors=True)
        return
    jobs.mark(key)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            inputs = lab.prepare_inputs(str(p / meta["photo"]))
            m = trimesh.load(str(p / meta["model"]), force="mesh", process=False)
            h = float(m.vertices[:, 1].max() - m.vertices[:, 1].min())
            V = m.vertices * (lab.HEIGHT_MM / h) if h > 0 else m.vertices
            t0 = time.time()
            metrics, _, _ = lab.score(meta.get("code") or lab.CODE_PRODUCTION, BodyMesh(vertices=V, faces=m.faces), inputs)
        row = ledger.make_row("app", meta["counter"], meta.get("code") or f"app:{meta.get('method')}", ledger.photo_sha(p / meta["photo"]),
                              metrics, job_id=job_id, method=meta.get("method"), tweaks=meta.get("tweaks"),
                              height_mm=meta.get("height_mm"), at_size=meta.get("at_size"), pick=meta.get("pick"),
                              render_seconds=meta.get("render_seconds"), measure_seconds=round(time.time() - t0, 1))
        row["at"] = meta.get("at", row["at"])
        # the wall as the print is made (the export's own measure at the asked height), not at the Lab's 150 mm
        w = (meta.get("at_size") or {}).get("wall_mm")
        if w is not None:
            row["R"]["wall"] = float(w)
        if (meta.get("at_size") or {}).get("watertight") is not None:
            row["R"]["sealed"] = 1.0 if meta["at_size"]["watertight"] else 0.0
        ledger.append(row)
    except Exception as exc:  # noqa: BLE001
        ledger.append({"source": "app", "counter": meta.get("counter"), "job_id": job_id, "at": meta.get("at"), "commit": ledger.machine(),
                       "code": meta.get("code"), "status": "unmeasured", "error": f"{type(exc).__name__}: {str(exc)[:200]}", "R": {}})
    finally:
        shutil.rmtree(p, ignore_errors=True)
        jobs.clear()


def production_like(code: str) -> bool:
    """True when the code is the app's own production build (option 0
    everywhere, but for the outline's "below the photo", which follows the
    photo) -- the app builds that with its own pipeline."""
    try:
        fam, idx = lab.decode(code)
    except ValueError:
        return False
    legs = [d[0] for d in lab.FAMILIES[fam]["dims"]].index("legs") if fam == "O" else -1
    return fam in ("O", "C") and all(i == 0 for k, i in enumerate(idx) if k != legs)


def build_pick(job_dir: str) -> int:
    """An app render built with the production pick (lab.build), written as
    the app writes its own (model.obj, model.glb, model.stl at the asked
    height). Runs in a process of its own. 0 when it built."""
    from anatomy3d.mesh_export import export_glb, export_obj, to_trimesh
    from anatomy3d.mesh_types import BodyMesh
    from anatomy3d.print_prep import export_stl

    jd = Path(job_dir)
    req = _read_json(jd / "pick_request.json", None)
    if req is None:
        return 3
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            inputs = lab.prepare_inputs(req["photo"])
            code = req["code"]
            fam, idx = lab.decode(code)
            if fam == "O":
                # "below the photo" follows this photo, not the held one
                legs = [d[0] for d in lab.FAMILIES["O"]["dims"]].index("legs")
                idx = list(idx)
                idx[legs] = 1 if inputs.bust_hint else 0
                code = lab.encode("O", tuple(idx))
            why = lab.refused(code, inputs.animal_frac)
            if why:
                raise RuntimeError(why)
            body = lab.build(code, inputs)
            s = float(req["height_mm"]) / lab.HEIGHT_MM
            out = BodyMesh(vertices=body.vertices * s, faces=body.faces, colors=getattr(body, "colors", None))
            export_obj(out, str(jd / "model.obj"))
            export_glb(out, str(jd / "model.glb"))
            export_stl(to_trimesh(out), str(jd / "model.stl"), target_faces=20000)
        lines = [x for x in said.getvalue().splitlines() if x.strip() and ("Wall" in x or "Warning" in x or "watertight" in x)]
        _write_json(jd / "pick_done.json", {"code": code, "messages": lines})
        return 0
    except Exception as exc:  # noqa: BLE001 - the app falls back to its own build
        _write_json(jd / "pick_done.json", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return 1


@router.get("/ledger")
def ledger_summary():
    """What the ledger holds and what it decided: the held job, the latest
    guard, the production pick, and the counts."""
    rs = ledger.rows()
    guards = sorted(ledger.root().glob("guard_*.json"), key=lambda q: q.stat().st_mtime)
    h = ledger.held()
    return {
        "rows": len(rs), "app_renders": sum(1 for r in rs if r.get("source") == "app"),
        "lab_builds": sum(1 for r in rs if r.get("source") == "lab"), "ratings": sum(1 for r in rs if r.get("source") == "rating"),
        "held": {"photo": h.get("photo"), "settings": len(h.get("codes") or [])},
        "guard": _read_json(guards[-1], None) if guards else None,
        "pick": ledger.pick(), "commit": ledger.commit(), "machine": ledger.machine(), "pending_measures": len(_pending_measures()),
    }


@router.get("/ledger/reading")
def ledger_reading(q: str, family: str = "O"):
    """The plugin's reading of one quantity over the held job, a window per commit."""
    if q not in ledger.QUANTITIES or family not in lab.FAMILIES:
        raise HTTPException(404, "Unknown quantity or family.")
    got, ref = ledger.read_held(q, family)
    if got is None:
        return {"q": q, "family": family, "reading": None, "why": ref}
    return {"q": q, "family": family, "reading": got, "reference": ref}


@router.get("/ledger/guard/{commit}")
def ledger_guard(commit: str):
    g = _read_json(ledger.root() / f"guard_{commit.replace(':', '_')}.json", None)
    if g is None:
        raise HTTPException(404, "No guard for this commit yet.")
    return g


def _worker_loop() -> None:
    turn = 0
    try:
        _ensure_calibration()
    except Exception as exc:  # noqa: BLE001 - the Lab goes on without it
        print(f"Ledger: calibration not started: {exc}")
    last_machine = ledger.machine()
    while True:
        run_id = None
        try:
            turn += 1
            housekeeping()
            if turn % 20 == 0 and ledger.machine() != last_machine:
                # tuned values changed (the Study's autopilot): a new machine, so the held job again
                last_machine = ledger.machine()
                _ensure_calibration()
            pend = _pending_measures()
            if pend:
                # an app render's reading first: it's one build's worth, and the person may rate it
                _run_job("measure", pend[0])
                time.sleep(0.3)
                continue
            sid = study_api.active()
            if sid is None and study_api.autopilot_step(_generation_lock):
                _release_memory()
                continue
            run_id = _active_run()
            if sid and (run_id is None or turn % 3):  # a running study gets two turns to a run's one
                run_id = None
                _run_job("study", sid)
                time.sleep(0.3)
                continue
            if run_id is None:
                time.sleep(3)
                continue
            _run_job("run", run_id)
            time.sleep(0.3)  # let a person's own Generate take the lock between combinations
        except Exception as exc:  # noqa: BLE001 - keep the worker alive whatever happens
            print(f"Lab worker error on {run_id or 'a study'}: {exc}")
            if run_id:
                with _state_lock:
                    d = LAB_ROOT / run_id
                    meta = _read_json(d / "meta.json", {})
                    meta.update(status="error", error=str(exc)[:300])
                    _write_json(d / "meta.json", meta)
            time.sleep(5)


def start_worker(generation_lock: threading.Lock) -> None:
    global _generation_lock
    _generation_lock = generation_lock
    threading.Thread(target=_worker_loop, name="lab-worker", daemon=True).start()


# ---------------------------------------------------------------- API


@router.get("/space")
def space():
    fams = []
    for key, fam in lab.FAMILIES.items():
        fams.append({
            "key": key,
            "name": fam["name"],
            "count": lab.family_size(key),
            "dims": [{"key": k, "label": label, "options": [o[1] for o in opts]} for k, label, opts in fam["dims"]],
        })
    params = [{"key": k, "label": label, "kind": kind, "how": how, "held_rule": rule} for k, label, kind, how, rule in lab.PARAMS]
    views = len(lab.FINISHES) * len(lab.SHADING)
    return {
        "params": params,
        "families": fams,
        "total_builds": lab.total_builds(),
        "finishes": [f[1] for f in lab.FINISHES],
        "shading": [s[1] for s in lab.SHADING],
        "total_views": lab.total_builds() * views,
        "templates": [{"name": n, "code": lab._template_code(f, o)} for n, f, o in lab.TEMPLATES],
        "score_formula": lab.SCORE_FORMULA,
    }


@router.post("/runs")
async def create_run(image: UploadFile = File(...)):
    run_id = uuid.uuid4().hex[:12]
    d = LAB_ROOT / run_id
    d.mkdir(parents=True)
    suffix = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        suffix = ".jpg"
    (d / f"photo{suffix}").write_bytes(await image.read())
    now = time.time()
    with _state_lock:
        for other in LAB_ROOT.iterdir():
            m = _read_json(other / "meta.json", None) if other.is_dir() else None
            if m and m.get("status") == "running":
                m["status"] = "paused"
                _write_json(other / "meta.json", m)
        _write_json(d / "order.json", lab.run_order(["C", "O", "B"], bust=False))
        _write_json(d / "meta.json", {
            "run_id": run_id, "photo_name": image.filename or "photo", "photo_file": f"photo{suffix}",
            "created": now, "started": now, "status": "running", "families": ["C", "O", "B"],
            "commit": ledger.machine(), "photo_sha": ledger.photo_sha(d / f"photo{suffix}"),
        })
    return {"run_id": run_id}


def _summary(d: Path) -> dict:
    meta = _read_json(d / "meta.json", {})
    order = _read_json(d / "order.json", [])
    results = _results(d)
    planned = [c for c in order if c.split("-")[0] in meta.get("families", ["C", "O"])]
    ok = [r for r in results.values() if r["status"] == "ok"]
    secs = [r["seconds"] for r in list(results.values())[-40:]]
    remaining = len(planned) - len(results)
    avg = (sum(secs) / len(secs)) if secs else None
    return {
        **{k: meta.get(k) for k in ("run_id", "photo_name", "status", "created", "families", "capsule_unavailable", "error")},
        "total": len(planned),
        "done": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "eta_hours": round(remaining * (avg + 0.3) / 3600, 1) if avg else None,
        "photo_url": f"/api/lab/runs/{meta.get('run_id')}/photo",
        "rated": len(_read_json(d / "ratings.json", {})),
    }


@router.get("/runs")
def list_runs():
    runs = []
    for d in sorted(LAB_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "meta.json").exists():
            runs.append(_summary(d))
    return {"runs": runs, "storage_gb": round(_folder_bytes() / 1e9, 2), "storage_cap_gb": round(LAB_MAX_BYTES / 1e9, 1)}


def _item(run_id: str, r: dict, ratings: dict) -> dict:
    code = r["code"]
    fam, idx = lab.decode(code)
    changed = [
        f"{label}: {opts[i][1]}" for (key, label, opts), i in zip(lab.FAMILIES[fam]["dims"], idx) if i != 0
    ]
    item = {
        "code": code, "n": r["n"], "family": lab.FAMILIES[fam]["name"], "template": lab.template_name(code),
        "changed": changed, "options": [f"{a}: {b}" for a, b in lab.describe(code)], "status": r["status"],
        "rating": ratings.get(code),
    }
    if r["status"] == "ok":
        item.update({k: r.get(k) for k in ("score", "shape_iou", "spike_pct", "watertight", "min_wall_mm", "face_found", "face_err_pct", "seconds", "params", "held_mean", "photo_detail", "checks_n", "checks_failed", "clothes_pct", "garment_edges", "garment_step_mm", "verified")})
        item["thumb_url"] = f"/api/lab/runs/{run_id}/thumb/{code}.png"
        item["model_url"] = f"/api/lab/runs/{run_id}/model/{code}.obj" if r.get("model") else None
    else:
        item["error"] = r.get("error")
    return item


@router.get("/runs/{run_id}")
def get_run(run_id: str, sort: str = "score", family: str = "all", offset: int = 0, limit: int = 40, only: str = "all",
            min_score: Optional[float] = None):
    """`min_score`: only combinations scoring at least this show; by
    default what the production combination scored on this photo (nothing
    below it is worth a look). Pass a very low number to see everything."""
    d = _run_dir(run_id)
    ratings = _read_json(d / "ratings.json", {})
    all_rows = list(_results(d).values())
    prod = next((r for r in all_rows if r["code"] == lab.CODE_PRODUCTION and r["status"] == "ok"), None)
    floor = min_score if min_score is not None else (prod.get("score") if prod else None)
    if only == "failed":
        floor = None  # the failures themselves, unscored
    rows = [r for r in all_rows if floor is None or (r["status"] == "ok" and (r.get("score") or -1e9) >= floor)
            or r["code"] in ratings]
    if family in ("C", "O", "B"):
        rows = [r for r in rows if r["code"].startswith(family + "-")]
    if only == "rated":
        rows = [r for r in rows if r["code"] in ratings]
    elif only == "failed":
        rows = [r for r in rows if r["status"] != "ok"]
    elif only == "ok":
        rows = [r for r in rows if r["status"] == "ok"]
    elif only == "verified":
        # sealed, printable, and the photo's clothes recognised on it (lab.score)
        rows = [r for r in rows if r["status"] == "ok" and r.get("verified")]
    if sort == "score":
        rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("score") or -1e9)))
    elif sort == "held":
        rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("held_mean") or -1e9)))
    elif sort == "rating":
        rows.sort(key=lambda r: (-(ratings.get(r["code"], {}).get("stars") or 0), -(r.get("score") or -1e9)))
    elif sort == "newest":
        rows.sort(key=lambda r: -r["at"])
    elif sort == "number":
        rows.sort(key=lambda r: r["n"])
    offset, limit = max(0, offset), max(1, min(limit, 200))
    return {
        "run": _summary(d),
        "count": len(rows),
        "min_score": floor,
        "production_score": prod.get("score") if prod else None,
        "hidden": len(all_rows) - len([r for r in all_rows if floor is None or (r["status"] == "ok" and (r.get("score") or -1e9) >= floor)
                                       or r["code"] in ratings]),
        "items": [_item(run_id, r, ratings) for r in rows[offset:offset + limit]],
    }


class Rating(BaseModel):
    code: str
    stars: int
    note: str = ""


@router.post("/runs/{run_id}/rate")
def rate(run_id: str, rating: Rating):
    d = _run_dir(run_id)
    if not CODE_RE.match(rating.code):
        raise HTTPException(400, "Bad combination code.")
    lab.decode(rating.code)
    with _state_lock:
        ratings = _read_json(d / "ratings.json", {})
        if rating.stars <= 0 and not rating.note.strip():
            ratings.pop(rating.code, None)
        else:
            ratings[rating.code] = {"stars": max(0, min(5, rating.stars)), "note": rating.note[:500], "at": time.time()}
        _write_json(d / "ratings.json", ratings)
    if rating.stars > 0:
        ledger.rate("lab", run_id, rating.code, max(1, min(5, rating.stars)), rating.note)
    return {"ok": True, "rated": len(ratings)}


@router.post("/runs/{run_id}/{action}")
def control_run(run_id: str, action: str):
    if action == "drop-models":
        return drop_models(run_id)
    if action not in ("pause", "resume"):
        raise HTTPException(404, "Unknown action.")
    d = _run_dir(run_id)
    with _state_lock:
        if action == "resume":
            for other in LAB_ROOT.iterdir():
                m = _read_json(other / "meta.json", None) if other.is_dir() else None
                if m and m.get("status") == "running" and other.name != run_id:
                    m["status"] = "paused"
                    _write_json(other / "meta.json", m)
        meta = _read_json(d / "meta.json", {})
        if meta.get("status") != "done":
            meta["status"] = "running" if action == "resume" else "paused"
            meta["started"] = time.time() if action == "resume" else meta.get("started")
            _write_json(d / "meta.json", meta)
    return _summary(d)


@router.get("/runs/{run_id}/export")
def export(run_id: str):
    d = _run_dir(run_id)
    ratings = _read_json(d / "ratings.json", {})
    rows = sorted(_results(d).values(), key=lambda r: r["n"])
    return JSONResponse({
        "run": _summary(d),
        "score_formula": lab.SCORE_FORMULA,
        "ratings": [
            {"code": c, "n": lab.number(c), "options": lab.describe(c), **v} for c, v in sorted(ratings.items(), key=lambda kv: -kv[1]["stars"])
        ],
        "results": [{**r, "options": lab.describe(r["code"])} for r in rows],
    })


@router.get("/runs/{run_id}/photo")
def photo(run_id: str):
    d = _run_dir(run_id)
    meta = _read_json(d / "meta.json", {})
    return FileResponse(d / meta["photo_file"])


@router.get("/runs/{run_id}/thumb/{code}.png")
def thumb(run_id: str, code: str):
    d = _run_dir(run_id)
    if not CODE_RE.match(code) or not (d / "t" / f"{code}.png").exists():
        raise HTTPException(404, "No thumbnail.")
    return FileResponse(d / "t" / f"{code}.png", media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/runs/{run_id}/model/{code}.obj")
def model(run_id: str, code: str):
    d = _run_dir(run_id)
    p = d / "m" / f"{code}.obj.gz"
    if not CODE_RE.match(code) or not p.exists():
        raise HTTPException(404, "No stored model for this combination.")
    return FileResponse(p, media_type="text/plain", headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=86400"})


@router.get("/runs/{run_id}/stl/{code}")
async def stl(run_id: str, code: str):
    """Full-quality printable STL, rebuilt on demand (previews are light)."""
    d = _run_dir(run_id)
    if not CODE_RE.match(code):
        raise HTTPException(400, "Bad combination code.")
    lab.decode(code)
    out = d / "stl"
    out.mkdir(exist_ok=True)
    path = out / f"{code}.stl"

    def make():
        from anatomy3d.mesh_export import to_trimesh
        from anatomy3d.print_prep import export_stl

        with _generation_lock, contextlib.redirect_stdout(io.StringIO()):
            body = lab.build(code, _inputs(run_id, d))
            export_stl(to_trimesh(body), str(path), target_faces=20000)

    if not path.exists():
        await run_in_threadpool(make)
    return FileResponse(path, media_type="model/stl", filename=f"figure-{code}.stl")


# ---------------------------------------------------------------- storage


def _bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


@router.get("/storage")
def storage():
    """What each run keeps on the volume: 3D previews (m/), thumbnails (t/)."""
    runs = []
    for d in sorted(LAB_ROOT.iterdir(), key=lambda p: p.stat().st_mtime):
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta:
            runs.append({"run_id": d.name, "photo_name": meta.get("photo_name"), "status": meta.get("status"),
                         "created": meta.get("created"), "models_mb": round(_bytes(d / "m") / 1e6, 1),
                         "thumbs_mb": round(_bytes(d / "t") / 1e6, 1), "rated": len(_read_json(d / "ratings.json", {}))})
    return {"runs": runs, "free_mb": round(_disk_free() / 1e6), "cap_gb": round(LAB_MAX_BYTES / 1e9, 1)}


def drop_models(run_id: str):
    """Removes a run's stored 3D previews (its scores, thumbnails and your
    ratings stay; Download STL still rebuilds any combination)."""
    d = _run_dir(run_id)
    meta = _read_json(d / "meta.json", {})
    if meta.get("status") == "running":
        raise HTTPException(409, "Pause the run first.")
    freed = _bytes(d / "m")
    shutil.rmtree(d / "m", ignore_errors=True)
    with _state_lock:
        rows = list(_results(d).values())
        for r in rows:
            r["model"] = False
        tmp = d / "results.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(d / "results.jsonl")
    _lab_bytes["total"] = None
    return {"freed_mb": round(freed / 1e6, 1), "free_mb": round(_disk_free() / 1e6)}


# ---------------------------------------------------------------- housekeeping
# The server clears out what nobody needs, on its own, so the volume never
# fills: rebuilt STLs after a day, study input caches once a study stops,
# and -- while the volume has under KEEP_FREE left or the Lab's folder is
# over its cap -- the 3D previews of runs that aren't running, oldest first,
# then their thumbnails (never a thumbnail you rated). Scores, ratings,
# photos, studies' results and tuned.json are never touched. Everything
# removed is logged to housekeeping.jsonl next to the Lab's folder.

KEEP_FREE = float(os.environ.get("ANATOMY3D_KEEP_FREE_GB", "1")) * 1e9
STL_TTL = 24 * 3600
HOUSEKEEPING_EVERY = 600
_housekeeping = {"last": 0.0}


def _housekeeping_log() -> Path:
    return LAB_ROOT.parent / "housekeeping.jsonl"


def _note(what: str, freed: int) -> None:
    if freed <= 0:
        return
    try:
        with open(_housekeeping_log(), "a") as f:
            f.write(json.dumps({"at": time.time(), "what": what, "freed_mb": round(freed / 1e6, 1)}) + "\n")
    except OSError:
        pass


def housekeeping(force: bool = False) -> None:
    now = time.time()
    if not force and now - _housekeeping["last"] < HOUSEKEEPING_EVERY:
        return
    _housekeeping["last"] = now
    runs = []
    for d in LAB_ROOT.iterdir():
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta:
            runs.append((meta.get("created", 0), d, meta))
    runs.sort(key=lambda r: r[0])
    for _, d, _ in runs:
        old = [p for p in (d / "stl").glob("*.stl") if now - p.stat().st_mtime > STL_TTL] if (d / "stl").is_dir() else []
        freed = sum(p.stat().st_size for p in old)
        for p in old:
            p.unlink(missing_ok=True)
        _note(f"rebuilt STLs of run {d.name}", freed)
    study_api.clear_idle_caches()

    def short() -> bool:
        return _disk_free() < KEEP_FREE or _folder_bytes() > LAB_MAX_BYTES

    idle = [(d, meta) for _, d, meta in runs if meta.get("status") != "running"]
    for d, _ in idle:
        if not short():
            break
        if (d / "m").is_dir():
            freed = _bytes(d / "m")
            drop_models(d.name)
            _note(f"3D previews of run {d.name}", freed)
    for d, _ in idle:
        if not short():
            break
        if (d / "t").is_dir():
            rated = set(_read_json(d / "ratings.json", {}))
            freed = 0
            for p in (d / "t").glob("*.png"):
                if p.stem not in rated:
                    freed += p.stat().st_size
                    p.unlink(missing_ok=True)
            _lab_bytes["total"] = None
            _note(f"unrated thumbnails of run {d.name}", freed)


@router.get("/housekeeping")
def housekeeping_log(limit: int = 30):
    p = _housekeeping_log()
    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return {"removed": lines[-max(1, min(limit, 200)):][::-1], "keep_free_gb": round(KEEP_FREE / 1e9, 1),
            "free_mb": round(_disk_free() / 1e6)}
