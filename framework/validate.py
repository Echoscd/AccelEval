"""
validate.py - ORBench v2 correctness validation

Compares:
  output.txt vs expected_output.txt
Text format: each line is a space-separated list of V floats (distance array).
Compares float-by-float with tolerance (atol) for correctness.
No top-k or checksum computation overhead.
"""

import os
import struct
import subprocess
import numpy as np
from dataclasses import dataclass, field

from .task import load_task, get_task_dir


@dataclass
class ValidationResult:
    correct: bool = False
    results_by_size: dict = field(default_factory=dict)  # size_name -> bool
    error: str = ""


def run_program(exe_path: str, args: list[str] = None, timeout: int = 180,
                env: dict = None, cwd: str = None) -> tuple[bool, str, str]:
    """
    Run a compiled executable, return (success, stdout, stderr).
    """
    cmd = [exe_path] + (args or [])
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
            env=run_env,
            cwd=cwd,
        )
        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)


def generate_test_data(task_id: str, size_name: str, output_dir: str) -> bool:
    """
    Run tasks/<task_id>/gen_data.py to produce v2 inputs for a given size.
    It should create:
      input.bin, requests.txt, expected_output.txt, cpu_time_ms.txt
    """
    task_dir = get_task_dir(task_id)
    gen_script = os.path.join(task_dir, "gen_data.py")
    if not os.path.exists(gen_script):
        return False
    os.makedirs(output_dir, exist_ok=True)
    try:
        result = subprocess.run(
            ["python3", gen_script, size_name, output_dir],
            capture_output=True,
            timeout=300,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
        return result.returncode == 0
    except Exception:
        return False


def data_exists(data_dir: str) -> bool:
    required = ["input.bin", "requests.txt", "expected_output.txt"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


def validate_solution(
    task_id: str,
    gpu_exe_path: str,
    device_id: int = 0,
) -> ValidationResult:
    """
    Validate a GPU solution against pre-generated expected output (v2).

    Flow:
      1. Ensure data exists (or run gen_data.py)
      2. Run GPU executable with --validate → writes output.txt
      3. Compare output.txt vs expected_output.txt (text format: space-separated floats)
    """
    task = load_task(task_id)
    task_dir = get_task_dir(task_id)
    result = ValidationResult()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    sizes_to_test = task.input_sizes if task.input_sizes else {"default": {}}
    # Optional: restrict validation sizes to avoid extremely slow expected generation (e.g. large graphs)
    # Example:
    #   ORBENCH_VALIDATE_SIZES=small python ...
    restrict = os.environ.get("ORBENCH_VALIDATE_SIZES")
    if restrict:
        allow = {s.strip() for s in restrict.split(",") if s.strip()}
        sizes_to_test = {k: v for k, v in sizes_to_test.items() if k in allow}
        if not sizes_to_test:
            result.correct = False
            result.error = f"No sizes match ORBENCH_VALIDATE_SIZES={restrict}"
            return result

    for size_name, size_params in sizes_to_test.items():
        # 1. Check pre-generated data exists
        data_dir = os.path.join(task_dir, "data", size_name)
        if not data_exists(data_dir):
            # Avoid auto-generating expected for very large sizes (can be extremely slow).
            result.results_by_size[size_name] = False
            result.error += (
                f"Missing data for size '{size_name}'. "
                f"Expected files: input.bin, requests.txt, expected_output.txt. "
                f"Run: python3 tasks/{task_id}/gen_data.py {size_name} {data_dir} --with-expected. "
            )
            continue

        # 2. Get num_requests from task data
        num_requests = int(size_params.get("num_requests", 0))
        if num_requests <= 0:
            result.results_by_size[size_name] = False
            result.error += f"Invalid num_requests for size '{size_name}'. "
            continue

        # 3. Run GPU solution with --validate (writes output.txt)
        output_txt = os.path.join(data_dir, "output.txt")
        if os.path.exists(output_txt):
            os.remove(output_txt)

        gpu_ok, gpu_stdout, gpu_stderr = run_program(
            gpu_exe_path, args=[data_dir, "--validate"], timeout=task.timeout, env=env
        )
        if not gpu_ok:
            result.results_by_size[size_name] = False
            result.error += f"GPU solution failed on size '{size_name}': {gpu_stderr[:200]}. "
            continue

        # 4. Check output.txt was created
        if not os.path.exists(output_txt):
            result.results_by_size[size_name] = False
            result.error += f"GPU solution did not produce output.txt for size '{size_name}'. "
            continue

        # 5. Compare text files (one distance value per line)
        expected_txt = os.path.join(data_dir, "expected_output.txt")
        try:
            # Read actual output (one float per line)
            with open(output_txt, "r") as f:
                actual_distances = [float(line.strip()) for line in f.readlines() if line.strip()]
            # Read expected output
            with open(expected_txt, "r") as f:
                expected_distances = [float(line.strip()) for line in f.readlines() if line.strip()]
        except Exception as ex:
            result.results_by_size[size_name] = False
            result.error += f"Failed to read output/expected for '{size_name}': {ex}. "
            continue

        # Check line count matches
        if len(actual_distances) != num_requests or len(expected_distances) != num_requests:
            result.results_by_size[size_name] = False
            result.error += f"Line count mismatch on '{size_name}': {len(actual_distances)} vs {len(expected_distances)} (expected {num_requests}). "
            continue

        # Compare distance values
        atol = task.atol if task.atol is not None else 0.1
        passed = True
        msg = "PASS"
        
        for r in range(num_requests):
            actual_dist = actual_distances[r]
            expected_dist = expected_distances[r]
            
            # Check if distances match within tolerance
            if not np.isclose(actual_dist, expected_dist, atol=atol, equal_nan=True):
                diff = abs(actual_dist - expected_dist)
                passed = False
                msg = f"request {r}: distance diff {diff:.6e} > {atol} (got {actual_dist:.6e}, expected {expected_dist:.6e})"
                break

        result.results_by_size[size_name] = passed
        print(f"  [{size_name}] {msg}")
        if not passed:
            result.error += f"Output mismatch on size '{size_name}': {msg}. "

    result.correct = bool(result.results_by_size) and all(result.results_by_size.values())
    return result
