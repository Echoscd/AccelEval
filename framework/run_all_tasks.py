#!/usr/bin/env python3
"""
run_all_tasks.py - One-shot pipeline: generate → evaluate → analyze for all tasks.

Uses tqdm for progress tracking with concurrent execution.

Usage:
    # Run with specific models and default settings
    python -m framework.run_all_tasks --models claude-sonnet-4 gpt-4o --levels 2 --samples 3

    # Run all models, all tasks, skip cost confirmation
    python -m framework.run_all_tasks --yes

    # Custom GPU settings
    python -m framework.run_all_tasks --models deepseek-v3 --gpus 2 --arch sm_80 --sizes small
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required. Install with: pip install tqdm")
    sys.exit(1)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.task import load_all_tasks, ORBENCH_ROOT
from framework.config import load_config, set_config, get_config
from framework.llm.registry import LLMRegistry
from framework.llm.scheduler import GenerationScheduler, GenerationJob, estimate_cost
from framework.batch_eval import eval_single_sample, save_eval_result, EvalResult
from framework.analyze import compute_summary, print_summary


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Generate solutions (concurrent, tqdm)
# ═══════════════════════════════════════════════════════════════

def run_generate(
    registry: LLMRegistry,
    model_ids: list[str],
    task_ids: list[str],
    levels: list[int],
    num_samples: int,
    max_workers: int,
    temperature: float,
    yes: bool,
) -> list[str]:
    """
    Generate CUDA solutions for all model×task×level×sample combinations.

    Returns list of run_names that were generated (for subsequent eval).
    """
    scheduler = GenerationScheduler(registry, runs_dir=os.path.join(ORBENCH_ROOT, "runs"))
    jobs = scheduler.build_jobs(model_ids, task_ids, levels, num_samples)

    if not jobs:
        print("  No generation jobs to run.")
        return []

    # Cost estimate
    est = estimate_cost(registry, jobs)
    print(f"\n{'='*60}")
    print(f"  Phase 1: Generate Solutions")
    print(f"{'='*60}")
    print(f"  Models:  {model_ids}")
    print(f"  Tasks:   {len(task_ids)}")
    print(f"  Levels:  {levels}")
    print(f"  Samples: {num_samples}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Est. cost:  ${est:.2f}")
    print(f"{'='*60}\n")

    if not yes and est > 0.0:
        confirm = input("  Proceed with generation? [y/N] ")
        if confirm.strip().lower() != "y":
            print("  Generation skipped.")
            return []

    # Filter already-done jobs
    pending = [j for j in jobs if not scheduler._already_done(j)]
    skipped = len(jobs) - len(pending)
    if skipped > 0:
        print(f"  Skipping {skipped} already-generated samples (resume)")

    if not pending:
        print("  All samples already generated.")
        run_names = sorted(set(scheduler._run_name(j) for j in jobs))
        return run_names

    # Determine concurrency
    providers_in_batch = set()
    for job in pending:
        providers_in_batch.add(registry.get_model_config(job.model_id)["provider"])
    total_workers = min(len(providers_in_batch) * max_workers, len(pending))

    from framework.llm import logger as llm_logger
    from framework.llm.resilient import RateLimiter, ResilientLLMClient
    llm_logger.init_logger(tag="run_all")

    results = []
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        future_to_job = {
            pool.submit(scheduler._execute_one, job, temperature): job
            for job in pending
        }

        with tqdm(total=len(pending), desc="Generating", unit="sample") as pbar:
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                total_cost += result.cost_usd

                status = "OK" if result.success else f"FAIL"
                j = result.job
                pbar.set_postfix_str(
                    f"{j.model_id}/{j.task_id}/s{j.sample_id} {status} (${total_cost:.3f})"
                )
                pbar.update(1)

    success_count = sum(1 for r in results if r.success)
    print(f"\n  Generation done: {success_count}/{len(pending)} succeeded, ${total_cost:.4f} total\n")

    run_names = sorted(set(scheduler._run_name(j) for j in jobs))
    return run_names


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Evaluate solutions (concurrent, tqdm)
# ═══════════════════════════════════════════════════════════════

def run_eval(
    run_name: str,
    task_ids: list[str] = None,
    arch: str = None,
    num_gpus: int = None,
    timeout: int = None,
    sizes: list[str] = None,
) -> str:
    """
    Evaluate all samples in a run directory.

    Returns path to the eval results JSON file.
    """
    config = get_config()
    if arch is None:
        arch = config.gpu.arch
    if num_gpus is None:
        num_gpus = config.eval.num_gpu_devices
    if timeout is None:
        timeout = config.eval.timeout

    if sizes:
        os.environ["ORBENCH_VALIDATE_SIZES"] = ",".join(sizes)

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name)
    if not os.path.exists(run_dir):
        print(f"  [WARN] Run directory not found: {run_dir}")
        return ""

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    eval_file = os.path.join(run_dir, f"eval_results_{ts}.json")

    # Discover samples
    if task_ids is None:
        task_ids = [
            d for d in sorted(os.listdir(run_dir))
            if os.path.isdir(os.path.join(run_dir, d))
        ]

    work_list = []
    for task_id in task_ids:
        task_dir = os.path.join(run_dir, task_id)
        if not os.path.isdir(task_dir):
            continue
        for filename in sorted(os.listdir(task_dir)):
            if filename.startswith("sample_") and filename.endswith(".cu"):
                sample_id = int(filename.split("_")[1].split(".")[0])
                sample_path = os.path.join(task_dir, filename)
                work_list.append((task_id, sample_path, sample_id))

    if not work_list:
        print(f"  [WARN] No samples found in {run_name}")
        return ""

    print(f"  Evaluating {run_name}: {len(work_list)} samples on {num_gpus} GPU(s)")

    with tqdm(total=len(work_list), desc=f"  Eval {run_name}", unit="sample") as pbar:
        for i, (task_id, sample_path, sample_id) in enumerate(work_list):
            device_id = i % num_gpus
            try:
                result = eval_single_sample(
                    task_id, sample_path, sample_id,
                    device_id=device_id, arch=arch,
                    run_nsys=False,
                )
            except Exception as e:
                result = EvalResult(
                    task_id=task_id, sample_id=sample_id,
                    error=f"Error: {str(e)[:200]}",
                )

            save_eval_result(result, eval_file)

            status = []
            if result.compiled:
                status.append("compiled")
            if result.correct:
                status.append("pass")
            if result.benchmark and result.benchmark.get("speedup_e2e", -1) > 0:
                status.append(f"{result.benchmark['speedup_e2e']:.1f}x")
            pbar.set_postfix_str(f"{task_id}/s{sample_id} {' '.join(status)}")
            pbar.update(1)

    return eval_file


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Analyze & aggregate results
# ═══════════════════════════════════════════════════════════════

def run_analyze(run_names: list[str], output_dir: str) -> dict:
    """
    Analyze all runs and produce a combined summary.

    Returns the combined summary dict.
    """
    all_summaries = {}

    for run_name in run_names:
        try:
            summary = compute_summary(run_name)
            all_summaries[run_name] = summary
            print(f"\n  ── {run_name} ──")
            print_summary(summary)
        except FileNotFoundError:
            print(f"  [WARN] No eval results for {run_name}, skipping analysis.")

    if not all_summaries:
        print("  No results to analyze.")
        return {}

    # Build combined report
    combined = {
        "timestamp": datetime.now().isoformat(),
        "runs": {},
        "comparison": {},
    }

    for run_name, summary in all_summaries.items():
        combined["runs"][run_name] = summary

    # Cross-run comparison table (per task)
    all_task_ids = set()
    for summary in all_summaries.values():
        all_task_ids.update(summary.get("tasks", {}).keys())

    for task_id in sorted(all_task_ids):
        combined["comparison"][task_id] = {}
        for run_name, summary in all_summaries.items():
            ts = summary.get("tasks", {}).get(task_id)
            if ts:
                combined["comparison"][task_id][run_name] = {
                    "compile_rate": ts.get("compile_rate", 0),
                    "pass_rate": ts.get("pass_rate", 0),
                    "best_speedup_e2e": ts.get("best_speedup_e2e"),
                }

    # Save combined report
    os.makedirs(output_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"all_results_{ts_str}.json")
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Combined report saved to: {report_path}")
    print(f"{'='*60}\n")

    # Print cross-run comparison
    if len(all_summaries) > 1:
        _print_comparison(combined, list(all_summaries.keys()))

    return combined


def _print_comparison(combined: dict, run_names: list[str]):
    """Print a cross-run comparison table."""
    comparison = combined.get("comparison", {})
    if not comparison:
        return

    print(f"\n{'='*70}")
    print(f"  Cross-Run Comparison")
    print(f"{'='*70}\n")

    header = f"  {'Task':<25s}"
    for rn in run_names:
        short = rn[:15]
        header += f" {short:>15s}"
    print(header)
    print(f"  {'-'*(25 + 16 * len(run_names))}")

    for task_id in sorted(comparison.keys()):
        row = f"  {task_id:<25s}"
        for rn in run_names:
            data = comparison[task_id].get(rn)
            if data:
                sp = data.get("best_speedup_e2e")
                pr = data.get("pass_rate", 0)
                cell = f"{pr:.0%}"
                if sp and sp > 0:
                    cell += f"/{sp:.0f}x"
                row += f" {cell:>15s}"
            else:
                row += f" {'—':>15s}"
        print(row)

    print()


# ═══════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ORBench: Generate → Evaluate → Analyze (all-in-one)"
    )

    # Generation args
    parser.add_argument("--models", nargs="*", default=None,
        help="Model IDs from models.yaml (default: all)")
    parser.add_argument("--tasks", nargs="*", default=None,
        help="Task IDs (default: all)")
    parser.add_argument("--levels", type=int, nargs="*", default=[2],
        help="Prompt levels (default: [2])")
    parser.add_argument("--samples", type=int, default=3,
        help="Samples per model×task×level")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=3,
        help="Max concurrent workers per provider")
    parser.add_argument("--models-yaml", default=None,
        help="Path to models.yaml")
    parser.add_argument("--yes", action="store_true",
        help="Skip cost confirmation")

    # Eval args
    parser.add_argument("--arch", default=None, help="GPU architecture")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sizes", nargs="*", default=None,
        help="Data sizes to evaluate (e.g., small medium large)")

    # Pipeline control
    parser.add_argument("--skip-generate", action="store_true",
        help="Skip generation, only eval+analyze existing runs")
    parser.add_argument("--skip-eval", action="store_true",
        help="Skip eval, only analyze existing results")
    parser.add_argument("--runs", nargs="*", default=None,
        help="Explicit run names (for --skip-generate mode)")
    parser.add_argument("--output-dir", default=None,
        help="Directory for combined report (default: runs/reports)")

    args = parser.parse_args()

    # Load config
    cli_args = {}
    if args.arch:
        cli_args["arch"] = args.arch
    if args.gpus:
        cli_args["gpus"] = args.gpus
    if args.timeout:
        cli_args["timeout"] = args.timeout
    config = load_config(cli_args=cli_args)
    set_config(config)

    output_dir = args.output_dir or os.path.join(ORBENCH_ROOT, "runs", "reports")

    # Resolve task list
    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = [t.task_id for t in load_all_tasks()]

    # ── Phase 1: Generate ──
    run_names = args.runs or []

    if not args.skip_generate:
        registry = LLMRegistry(args.models_yaml)
        if args.models:
            model_ids = args.models
            for mid in model_ids:
                registry.get_model_config(mid)
        else:
            model_ids = registry.list_models()

        generated_runs = run_generate(
            registry=registry,
            model_ids=model_ids,
            task_ids=task_ids,
            levels=args.levels,
            num_samples=args.samples,
            max_workers=args.workers,
            temperature=args.temperature,
            yes=args.yes,
        )
        run_names = list(set(run_names + generated_runs))

    if not run_names:
        print("No runs to evaluate. Specify --runs or run generation first.")
        sys.exit(1)

    # ── Phase 2: Evaluate ──
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Evaluate Solutions")
        print(f"{'='*60}")
        print(f"  Runs: {run_names}\n")

        for run_name in tqdm(run_names, desc="Eval runs", unit="run"):
            run_eval(
                run_name=run_name,
                task_ids=None,
                arch=args.arch,
                num_gpus=args.gpus,
                timeout=args.timeout,
                sizes=args.sizes,
            )

    # ── Phase 3: Analyze ──
    print(f"\n{'='*60}")
    print(f"  Phase 3: Analyze Results")
    print(f"{'='*60}")

    combined = run_analyze(run_names, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
