import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import run as run_plan
import train as train_script


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MFC training grid with parallel workers.")
    parser.add_argument(
        "--env",
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "all"],
        required=True,
    )
    parser.add_argument("--seeds", type=run_plan.parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--budget-mode", choices=["fair", "manual"], default="fair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=None)
    parser.add_argument("--law-chart", choices=["gaussian", "mean"], default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--jobs-file", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_against_root(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return run_plan.ROOT / path


def normalize_paths(args):
    args.results_root = str(resolve_against_root(args.results_root))
    if args.jobs_file is not None:
        args.jobs_file = resolve_against_root(args.jobs_file)
    if args.logs_root is not None:
        args.logs_root = resolve_against_root(args.logs_root)


def train_args_for(job_spec, seed, args):
    return SimpleNamespace(
        env=job_spec["env"],
        algorithm=job_spec["algorithm"],
        perturbation=job_spec["perturbation"],
        eta=None,
        horizon=job_spec["horizon"],
        flow=job_spec["flow"],
        seed=seed,
        results_root=args.results_root,
    )


def log_path_for(output_dir, results_root, logs_root):
    try:
        relative = output_dir.relative_to(results_root)
    except ValueError:
        relative = Path(*output_dir.parts[1:]) if output_dir.is_absolute() else output_dir
    return logs_root / relative.parent / f"{relative.name}.log"


def build_records(args):
    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
    selected_envs = envs if args.env == "all" else [args.env]
    results_root = Path(args.results_root)
    logs_root = args.logs_root or results_root / "logs"
    records = []

    for env in selected_envs:
        for job_spec in run_plan.experiment_plan(env):
            for seed in args.seeds:
                train_args = train_args_for(job_spec, seed, args)
                output_dir = train_script.output_directory(train_args)
                records.append(
                    {
                        "command": run_plan.command_for(job_spec, seed, args),
                        "output_dir": output_dir,
                        "summary_path": output_dir / "summary.json",
                        "log_path": log_path_for(output_dir, results_root, logs_root),
                    }
                )

    return records


def write_jobs_file(records, jobs_file, resume):
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with jobs_file.open("w", encoding="utf-8") as file:
        for record in records:
            completed = record["summary_path"].exists()
            payload = {
                "status": "skipped" if resume and completed else "pending",
                "command": [str(value) for value in record["command"]],
                "output_dir": str(record["output_dir"]),
                "summary_path": str(record["summary_path"]),
                "log_path": str(record["log_path"]),
            }
            file.write(json.dumps(payload) + "\n")


def run_record(record, index, total, print_lock):
    command = [str(value) for value in record["command"]]
    log_path = record["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    with print_lock:
        print(f"[{index}/{total}] start {record['output_dir']} -> {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + json.dumps(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=run_plan.ROOT)
        elapsed = time.perf_counter() - started_at
        log.write(f"\nexit_code={result.returncode} elapsed_seconds={elapsed:.3f}\n")

    if result.returncode != 0:
        raise RuntimeError(f"Job failed with exit code {result.returncode}: {record['output_dir']} ({log_path})")

    with print_lock:
        print(f"[{index}/{total}] done  {record['output_dir']} ({elapsed:.1f}s)", flush=True)


def run_pending(pending, workers):
    print_lock = threading.Lock()
    indexed = [(index, record) for index, record in enumerate(pending, start=1)]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_record, record, index, len(pending), print_lock): record
            for index, record in indexed
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((record, error))
                with print_lock:
                    print(f"FAILED {record['output_dir']} ({record['log_path']}): {error}", flush=True)

    if failures:
        print("\nFailed jobs:", flush=True)
        for record, error in failures:
            print(f"- {record['output_dir']} ({record['log_path']}): {error}", flush=True)
        raise RuntimeError(f"{len(failures)} job(s) failed; see per-job logs above.")


def main():
    args = parse_args()
    normalize_paths(args)
    if args.baseline and args.no_baseline:
        raise ValueError("Use at most one of --baseline and --no-baseline.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    records = build_records(args)
    if not records:
        raise RuntimeError("The experiment plan produced zero jobs; check --env and --seeds.")

    resume = not args.no_resume
    jobs_file = args.jobs_file or Path(args.results_root) / "train_jobs.jsonl"
    write_jobs_file(records, jobs_file, resume)

    completed = [record for record in records if record["summary_path"].exists()]
    pending = [record for record in records if not (resume and record["summary_path"].exists())]

    print(f"Prepared {len(records)} jobs.")
    print(f"Jobs file: {jobs_file}")
    print(f"Completed jobs skipped: {len(completed) if resume else 0}")
    print(f"Pending jobs: {len(pending)}")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        return
    if not pending:
        print("Nothing to run; every job already has summary.json.")
        return

    run_pending(pending, args.workers)


if __name__ == "__main__":
    main()
