"""
breakdown.py — Pattern Breakdown Agent.

Given code_a and code_b with identified pattern changes from a DiffRecord,
generate intermediate code versions A → A1 → A2 → ... → B, where each step
applies one pattern change (or one causal chain) at a time.

This enables pattern-level ablation: measuring the individual contribution
of each optimization pattern to the overall speedup.

Usage:
    from framework.knowledge.breakdown import run_breakdown
    chain = run_breakdown(diff_record, kb, agent_model_id="...")

CLI:
    python -m framework.knowledge.cli breakdown <run_dir> --task <task_id> --diff <diff_id>
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class BreakdownStep:
    step_id: int
    description: str               # what this step applies
    pattern_ids: list[str]         # PAT-xxx or causal chain IDs
    is_causal_chain: bool          # True if this step applies a whole chain
    code: str                      # generated intermediate code
    source_path: str = ""          # path where code was saved
    compiled: bool = False
    correct: bool = False
    speedup_e2e: float = -1.0
    kernel_time_ms: float = -1.0
    eval_error: str = ""


@dataclass
class BreakdownResult:
    diff_id: str
    task_id: str
    code_a_path: str
    code_b_path: str
    speedup_a: float
    speedup_b: float
    steps: list[BreakdownStep]
    model_id: str
    timestamp: str


# ---------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """\
You are a CUDA optimization analyst. You will receive:
1. Version A code (slower or baseline)
2. Version B code (faster or modified)
3. A list of pattern changes identified between A and B
4. Causal chains (groups of interdependent changes)

Your job is to create a PLAN to incrementally transform A into B, one pattern
(or causal chain) at a time.

Rules:
- Each step applies exactly ONE independent pattern change, OR one entire causal
  chain (because the patterns in a chain are interdependent and cannot be separated).
- Order steps from most fundamental to most superficial. Infrastructure changes
  (data layout, precomputation) should come before compute changes (loop unroll,
  branchless). Changes that others depend on come first.
- The final step's result should be functionally equivalent to Version B.
- Every intermediate version must be COMPILABLE and CORRECT (produce the same output).

Output strict JSON (no markdown fences):
{
  "steps": [
    {
      "step_id": 1,
      "description": "Apply X: brief description of what to change",
      "pattern_ids": ["PAT-XXX"],
      "is_causal_chain": false,
      "depends_on_step": null
    },
    {
      "step_id": 2,
      "description": "Apply causal chain Y: change A which enables B which enables C",
      "pattern_ids": ["PAT-AAA", "PAT-BBB"],
      "is_causal_chain": true,
      "depends_on_step": 1
    }
  ],
  "rationale": "Why this ordering"
}
"""

PLAN_USER = """\
## Version A code ({speedup_a:.1f}x speedup):
```cuda
{code_a}
```

## Version B code ({speedup_b:.1f}x speedup):
```cuda
{code_b}
```

## Pattern changes between A and B:
{pattern_changes_summary}

## Causal chains:
{causal_chains_summary}
"""


STEP_SYSTEM = """\
You are a CUDA code transformation specialist. You will receive:
1. The CURRENT version of the code (starting from version A)
2. The TARGET version (version B) for reference
3. A specific transformation to apply

Your job is to apply ONLY the specified transformation to the current code,
producing a new version that:
- Is COMPILABLE (valid CUDA C++)
- Is CORRECT (produces the same numerical output as both A and B)
- Applies ONLY the specified change, keeping everything else the same as the current version
- Preserves the same function signatures (extern "C" void solution_compute(...) etc.)

IMPORTANT:
- Do NOT apply other optimizations from version B that are not part of this step.
- The code must compile and run correctly. If applying the change requires small
  supporting changes (e.g., adding a variable declaration), include those.
- Output ONLY the complete .cu file content. No explanations, no markdown fences.
"""

STEP_USER = """\
## Current code (after previous steps):
```cuda
{current_code}
```

## Target version B (for reference only — do NOT copy wholesale):
```cuda
{code_b}
```

## Transformation to apply in this step:
{step_description}

## Pattern IDs: {pattern_ids}

## What specifically changed in version B for this pattern:
{what_changed}

Output the complete transformed .cu file:
"""


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_source(path: str) -> str:
    if not path or not os.path.exists(path):
        return "(source not found)"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_json_response(text: str) -> dict:
    import re
    if not text:
        return {}
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return {}


def _extract_cuda_code(text: str) -> str:
    """Extract CUDA code from LLM response, stripping markdown fences if present."""
    import re
    text = text.strip()
    # Try to find code in markdown fences
    m = re.search(r'```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If starts with #include, it's raw code
    if text.startswith('#include') or text.startswith('//') or text.startswith('/*'):
        return text
    # Try to find the first #include
    idx = text.find('#include')
    if idx >= 0:
        return text[idx:]
    return text


def _format_pattern_changes(diff_data: dict) -> str:
    changes = diff_data.get("pattern_changes", [])
    if not changes:
        return "(no pattern changes)"
    lines = []
    for c in changes:
        pid = c.get("pattern_id", "?")
        name = c.get("pattern_name", "")
        ctype = c.get("change_type", "?")
        what = c.get("what_changed", "")
        lines.append(f"- {pid} ({name}): {ctype} — {what}")
    return "\n".join(lines)


def _format_causal_chains(diff_data: dict) -> str:
    chains = diff_data.get("causal_chains", [])
    if not chains:
        return "(no causal chains identified)"
    lines = []
    for cc in chains:
        name = cc.get("name", "?")
        pids = cc.get("pattern_ids", [])
        trigger = cc.get("trigger", "")
        steps = cc.get("steps", [])
        lines.append(f"- {name} [{', '.join(pids)}]: trigger={trigger}")
        for s in steps:
            lines.append(f"    → {s}")
    return "\n".join(lines)


def _get_what_changed_for_patterns(diff_data: dict, pattern_ids: list[str]) -> str:
    """Get what_changed descriptions for specific pattern IDs."""
    changes = diff_data.get("pattern_changes", [])
    parts = []
    for c in changes:
        if c.get("pattern_id") in pattern_ids:
            parts.append(f"{c.get('pattern_id')}: {c.get('what_changed', '')}")
            if c.get("code_b_evidence"):
                parts.append(f"  Code in B: {c['code_b_evidence'][:300]}")
    return "\n".join(parts) if parts else "(no specific details available)"


# ---------------------------------------------------------------------------
#  Core: run breakdown
# ---------------------------------------------------------------------------

def run_breakdown(
    diff_data: dict,
    code_a: str,
    code_b: str,
    task_id: str,
    kb,
    output_dir: str,
    agent_model_id: str = "gemini-3.1-pro-preview-openrouter",
    llm_client=None,
    run_eval: bool = True,
) -> BreakdownResult:
    """
    Run the full breakdown pipeline:
    1. Plan: decide step ordering
    2. Generate: produce intermediate code for each step
    3. (Optional) Eval: compile + benchmark each intermediate version
    """
    if llm_client is None:
        from ..llm.registry import LLMRegistry
        registry = LLMRegistry()
        llm_client = registry.get_client(agent_model_id)

    speedup_a = diff_data.get("version_a_speedup", 0)
    speedup_b = diff_data.get("version_b_speedup", 0)

    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: Plan ---
    print(f"[BREAKDOWN] Phase 1: Planning step order...")
    plan_user = PLAN_USER.format(
        speedup_a=speedup_a,
        speedup_b=speedup_b,
        code_a=code_a,
        code_b=code_b,
        pattern_changes_summary=_format_pattern_changes(diff_data),
        causal_chains_summary=_format_causal_chains(diff_data),
    )

    plan_resp = llm_client.generate(
        prompt=f"{PLAN_SYSTEM}\n\n{plan_user}",
        max_tokens=4000,
        temperature=0.0,
    )
    plan = _parse_json_response(plan_resp.content or "")

    if not plan or "steps" not in plan:
        print(f"[BREAKDOWN] WARNING: Could not parse plan. Raw: {(plan_resp.content or '')[:300]}")
        return BreakdownResult(
            diff_id=diff_data.get("diff_id", "?"),
            task_id=task_id,
            code_a_path=diff_data.get("version_a_path", ""),
            code_b_path=diff_data.get("version_b_path", ""),
            speedup_a=speedup_a,
            speedup_b=speedup_b,
            steps=[],
            model_id=agent_model_id,
            timestamp=_now_ts(),
        )

    planned_steps = plan["steps"]
    print(f"[BREAKDOWN] Plan: {len(planned_steps)} steps")
    for ps in planned_steps:
        print(f"  Step {ps['step_id']}: {ps['description'][:80]}")
    print(f"  Rationale: {plan.get('rationale', 'N/A')}")

    # Save plan
    with open(os.path.join(output_dir, "breakdown_plan.json"), "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    # --- Phase 2: Generate intermediate code ---
    print(f"\n[BREAKDOWN] Phase 2: Generating intermediate versions...")
    current_code = code_a
    breakdown_steps: list[BreakdownStep] = []

    for ps in planned_steps:
        step_id = ps["step_id"]
        pattern_ids = ps.get("pattern_ids", [])
        is_chain = ps.get("is_causal_chain", False)
        description = ps["description"]

        print(f"\n[BREAKDOWN] Step {step_id}: {description[:80]}...")

        what_changed = _get_what_changed_for_patterns(diff_data, pattern_ids)

        step_user = STEP_USER.format(
            current_code=current_code,
            code_b=code_b,
            step_description=description,
            pattern_ids=", ".join(pattern_ids),
            what_changed=what_changed,
        )

        step_resp = llm_client.generate(
            prompt=f"{STEP_SYSTEM}\n\n{step_user}",
            max_tokens=16000,
            temperature=0.0,
        )
        generated_code = _extract_cuda_code(step_resp.content or "")

        if not generated_code or len(generated_code) < 50:
            print(f"  WARNING: Generated code too short ({len(generated_code)} chars), skipping")
            breakdown_steps.append(BreakdownStep(
                step_id=step_id,
                description=description,
                pattern_ids=pattern_ids,
                is_causal_chain=is_chain,
                code="",
                eval_error="Generated code too short",
            ))
            continue

        # Save intermediate .cu file
        cu_path = os.path.join(output_dir, f"step_{step_id}.cu")
        with open(cu_path, "w") as f:
            f.write(generated_code)

        step = BreakdownStep(
            step_id=step_id,
            description=description,
            pattern_ids=pattern_ids,
            is_causal_chain=is_chain,
            code=generated_code,
            source_path=cu_path,
        )

        # --- Phase 3 (per step): Eval ---
        if run_eval:
            print(f"  Evaluating step {step_id}...")
            try:
                from ..batch_eval import eval_single_sample
                er = eval_single_sample(
                    task_id=task_id,
                    sample_path=cu_path,
                    sample_id=1000 + step_id,
                )
                step.compiled = er.compiled
                step.correct = er.correct
                if er.benchmark:
                    step.speedup_e2e = er.benchmark.get("speedup_e2e", -1.0)
                    step.kernel_time_ms = er.benchmark.get("kernel_time_ms", -1.0)
                if er.error:
                    step.eval_error = er.error[:300]

                status = "PASS" if step.correct else ("COMPILED" if step.compiled else "FAIL")
                spd = f"{step.speedup_e2e:.1f}x" if step.speedup_e2e > 0 else "N/A"
                print(f"  Step {step_id}: {status} speedup={spd}")

            except Exception as e:
                step.eval_error = str(e)[:300]
                print(f"  Step {step_id}: EVAL ERROR: {e}")

        breakdown_steps.append(step)

        # If step compiled and is correct, use it as the base for next step
        if step.compiled and step.correct:
            current_code = generated_code
        else:
            print(f"  WARNING: Step {step_id} failed eval, keeping previous code as base")

    # --- Save results ---
    result = BreakdownResult(
        diff_id=diff_data.get("diff_id", "?"),
        task_id=task_id,
        code_a_path=diff_data.get("version_a_path", ""),
        code_b_path=diff_data.get("version_b_path", ""),
        speedup_a=speedup_a,
        speedup_b=speedup_b,
        steps=breakdown_steps,
        model_id=agent_model_id,
        timestamp=_now_ts(),
    )

    result_path = os.path.join(output_dir, "breakdown_result.json")
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"\n[BREAKDOWN] Saved result to {result_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  BREAKDOWN SUMMARY: {task_id} {diff_data.get('diff_id','')}")
    print(f"  A={speedup_a:.1f}x → B={speedup_b:.1f}x ({speedup_b/max(speedup_a,0.001):.2f}x improvement)")
    print(f"{'='*70}")
    print(f"{'Step':<6} {'Patterns':<20} {'Status':<10} {'Speedup':>10} {'Kernel(ms)':>12} Description")
    print(f"{'-'*6} {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*30}")

    # Step 0 = original A
    print(f"{'A':<6} {'':<20} {'PASS':<10} {speedup_a:>10.1f}x {diff_data.get('version_a_kernel_time_ms',0):>12.2f} Original version A")

    for s in breakdown_steps:
        pids = ",".join(s.pattern_ids)[:18]
        status = "PASS" if s.correct else ("COMP" if s.compiled else "FAIL")
        spd = f"{s.speedup_e2e:.1f}x" if s.speedup_e2e > 0 else "N/A"
        kt = f"{s.kernel_time_ms:.2f}" if s.kernel_time_ms > 0 else "N/A"
        desc = s.description[:30]
        print(f"{s.step_id:<6} {pids:<20} {status:<10} {spd:>10} {kt:>12} {desc}")

    # Step B = target
    print(f"{'B':<6} {'':<20} {'PASS':<10} {speedup_b:>10.1f}x {diff_data.get('version_b_kernel_time_ms',0):>12.2f} Target version B")
    print()

    return result


# ---------------------------------------------------------------------------
#  CLI entry: load diff from file and run
# ---------------------------------------------------------------------------

def run_breakdown_from_diffs(
    diffs_path: str,
    diff_id: str | None,
    task_id: str | None,
    kb,
    agent_model_id: str = "gemini-3.1-pro-preview-openrouter",
    run_eval: bool = True,
) -> list[BreakdownResult]:
    """Load diffs from file, optionally filter, and run breakdown on each."""
    diffs = []
    with open(diffs_path) as f:
        if diffs_path.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    diffs.append(json.loads(line))
        else:
            diffs = json.load(f)
            if not isinstance(diffs, list):
                diffs = [diffs]

    # Filter
    if diff_id:
        diffs = [d for d in diffs if d.get("diff_id") == diff_id]
    if task_id:
        diffs = [d for d in diffs if d.get("task_id") == task_id]

    if not diffs:
        print(f"[BREAKDOWN] No matching diffs found (diff_id={diff_id}, task_id={task_id})")
        return []

    results = []
    for d in diffs:
        did = d.get("diff_id", "unknown")
        tid = d.get("task_id", "unknown")

        code_a = d.get("code_a") or _read_source(d.get("version_a_path", ""))
        code_b = d.get("code_b") or _read_source(d.get("version_b_path", ""))

        if "(source not found)" in code_a or "(source not found)" in code_b:
            print(f"[BREAKDOWN] Skipping {did}: source not found")
            continue

        output_dir = os.path.join(
            os.path.dirname(diffs_path), "breakdowns", f"{tid}_{did}"
        )

        print(f"\n{'#'*70}")
        print(f"# BREAKDOWN: {tid} {did}")
        print(f"# {d.get('version_a_id','')} ({d.get('version_a_speedup',0):.1f}x) → "
              f"{d.get('version_b_id','')} ({d.get('version_b_speedup',0):.1f}x)")
        print(f"{'#'*70}")

        r = run_breakdown(
            diff_data=d,
            code_a=code_a,
            code_b=code_b,
            task_id=tid,
            kb=kb,
            output_dir=output_dir,
            agent_model_id=agent_model_id,
            run_eval=run_eval,
        )
        results.append(r)

    return results
