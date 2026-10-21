"""Replay saved-result audits without overwriting archived audit reports."""
from pathlib import Path
from unittest.mock import patch
import contextlib
import json
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "verification"
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
AUDITS = [
    (RUN / "independent_audit.py", ["--run-dir", str(RUN)]),
    *[(ROOT / name, []) for name in [
        "independent_confirmation_audit.py", "audit_fake_backend.py",
        "audit_gate_folded_zne.py", "audit_fake_backend_noise_sweep.py",
        "audit_fake_backend_native_fold.py", "audit_standardized_depth_sweep.py",
        "audit_standardized_exact_noise_sweep.py", "audit_standardized_transfer.py",
    ]],
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_write = Path.write_text
    results = []
    for script, args in AUDITS:
        print(f"Running {script.name}", flush=True)
        started = time.monotonic()
        folder = OUT / script.stem
        folder.mkdir(exist_ok=True)
        def redirect(path, text, *a, **kw):
            # These audits write JSON reports only; preserve the original run snapshots.
            if path.suffix != ".json":
                raise RuntimeError(f"Unexpected audit output: {path.name}")
            return original_write(folder / path.name, text, *a, **kw)
        status, error = "PASS", None
        old_argv = sys.argv
        try:
            sys.argv = [str(script), *args]
            with (folder / "audit.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), patch.object(Path, "write_text", redirect):
                    runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                status, error = "FAIL", str(exc)
        except Exception as exc:
            status, error = "FAIL", repr(exc)
        finally:
            sys.argv = old_argv
        results.append({"audit": script.name, "status": status, "error": error,
                        "seconds": round(time.monotonic()-started, 2)})
        print(results[-1], flush=True)
    (OUT / "audit_summary.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    if any(r["status"] != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
