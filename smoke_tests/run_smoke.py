#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from genatator_core.utils import clean_env_for_gpu


def run_job(repo: Path, job: dict) -> int:
    cmd = job["command"]
    if int(job.get("num_processes", 1)) > 1:
        cmd = f"accelerate launch --num_processes {int(job['num_processes'])} {cmd}"
    log_dir = repo / "smoke_tests" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['name']}.log"
    env = clean_env_for_gpu(job.get("gpus"))
    print(f"[smoke] start {job['name']} on GPUs={job.get('gpus')} -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, shell=True, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        ret = proc.wait()
    print(f"[smoke] done {job['name']} exit={ret}")
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="smoke_tests/smoke_matrix.json")
    parser.add_argument("--only", nargs="*", default=None, help="Optional list of job names to run.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    matrix = json.loads((repo / args.matrix).read_text())
    if matrix.get("make_tiny_data", True):
        subprocess.check_call([sys.executable, "smoke_tests/make_tiny_data.py"], cwd=repo)
    jobs = [j for j in matrix["jobs"] if j.get("enabled", True)]
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j["name"] in wanted]
    max_parallel = int(matrix.get("max_parallel_jobs", 1))
    failures = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(run_job, repo, j): j for j in jobs}
        for fut in as_completed(futs):
            ret = fut.result()
            if ret != 0:
                failures.append(futs[fut]["name"])
    if failures:
        raise SystemExit(f"Smoke failed: {failures}")


if __name__ == "__main__":
    main()
