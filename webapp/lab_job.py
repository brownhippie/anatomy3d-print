"""One build of the Lab's worker, in a process of its own (lab_api._run_job).

    python lab_job.py run <run_id>     the run's next combination
    python lab_job.py study <sid>      the study's next step

A build peaks near 6 GB on some photos; in the server's own process the
service ran out of its 8 GB and was restarted mid-build, again and again,
retrying the same build each time. In a process of its own, all it took is
given back when it ends, and if it is killed only it goes: the server stays
up and the build is recorded as failed (jobs.py).
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab_api  # noqa: E402
import study_api  # noqa: E402


def main() -> int:
    kind, arg = sys.argv[1], sys.argv[2]
    # the approved study values, as the server has them, before any build
    study_api.load_tuned_at_startup()
    lock = threading.Lock()  # (this process's own: the server holds its lock while this one runs)
    lab_api._generation_lock = lock
    study_api.set_generation_lock(lock)
    if kind == "run":
        lab_api._work_one(arg)
    elif kind == "study":
        study_api.work_one(arg, lock)
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
