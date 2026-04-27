#!/usr/bin/env python3
"""
Stage 1 of the strategy-transfer pipeline.

For each task in runs/strategy_transfer/best/manifest.json whose winner
speedup is above --min-speedup:

  1. Run the static detector  (framework.knowledge.auto_detect.extract_auto_features)
  2. Match against the 43-pattern catalog
       (framework.knowledge.store.KnowledgeBase.match_by_features)
  3. Call the unified agent analyzer
       (framework.knowledge.agent_analyzer.analyze_sample)
     which produces structured per-pattern summaries covering both
     auto-detected patterns and manual-check patterns
  4. Render the JSON result into a natural-language recipe
       runs/strategy_transfer/guidance/<task>.md
     that another model can follow at Stage 2.

Usage:
    python3 scripts/generate_guidance.py
    python3 scripts/generate_guidance.py --only-tasks black_scholes
    python3 scripts/generate_guidance.py --force
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load API keys from .env (same convention as run.py)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from framework.llm.registry import LLMRegistry
from framework.knowledge.auto_detect import extract_auto_features
from framework.knowledge.store import KnowledgeBase
from framework.knowledge.agent_analyzer import analyze_sample

KB_DATA_DIR = ROOT / "Library" / "knowledge_data"


# ---------------------------------------------------------------------------
#  Render analyzer JSON -> guidance.md
# ---------------------------------------------------------------------------

def _fmt_intensity(p: dict) -> str:
    intensity = p.get("intensity_note") or p.get("intensity") or ""
    return f" (intensity: {intensity})" if intensity else ""


def render_guidance_md(
    task_id: str,
    task_desc: str,
    winner_model: str,
    winner_speedup: float,
    kb_num_patterns: int,
    analysis: dict,
) -> str:
    """Turn analyze_sample() output into a Stage-2-consumable markdown recipe."""
    strategy = (analysis.get("strategy_summary") or "").strip()
    bottleneck = (analysis.get("bottleneck_analysis") or "").strip()
    summaries = analysis.get("pattern_summaries") or []
    candidates = analysis.get("new_candidates") or []

    lines = []
    lines.append(f"# CUDA recipe: {task_id} ({winner_speedup:.1f}× winner)")
    lines.append("")

    if strategy:
        lines.append(f"**Overall strategy.** {strategy}")
        lines.append("")

    if summaries:
        lines.append(
            f"## Patterns used in the winning {winner_speedup:.1f}× solution "
            f"({len(summaries)} of {kb_num_patterns} catalog patterns matched)"
        )
        lines.append("")
        # Sort: auto_detected first, then alphabetical by pattern_id
        summaries_sorted = sorted(
            summaries,
            key=lambda x: (x.get("source", "") != "auto_detected", x.get("pattern_id", "")),
        )
        for i, p in enumerate(summaries_sorted, 1):
            pid = p.get("pattern_id", "?")
            name = p.get("pattern_name", "(unnamed)")
            src = p.get("source", "")
            tag = " (auto-detected)" if src == "auto_detected" else " (manual-check: verified)"
            lines.append(f"### {i}. `{pid}` {name}{_fmt_intensity(p)}{tag}")
            if tgt := (p.get("target") or "").strip():
                lines.append(f"- **Target:** {tgt}")
            if meth := (p.get("method") or "").strip():
                lines.append(f"- **Method:** {meth}")
            lines.append("")

    if candidates:
        lines.append("## Extra techniques not in the catalog")
        lines.append("")
        for j, c in enumerate(candidates, 1):
            desc = (c.get("raw_description") or "").strip()
            mech = (c.get("mechanism_hypothesis") or "").strip()
            impact = c.get("estimated_impact", "")
            # Use first sentence as heading; rest (if any) goes into description
            first_sent = desc.split(".")[0].strip()
            rest = desc[len(first_sent) + 1:].strip().lstrip(".")
            impact_tag = f" _(impact: {impact})_" if impact else ""
            lines.append(f"### {j}. {first_sent}{impact_tag}")
            if rest:
                lines.append(rest)
            if mech:
                lines.append(f"- **Why it helps:** {mech}")
            lines.append("")

    if bottleneck:
        lines.append("## Residual bottleneck (where further speedup must come from)")
        lines.append("")
        lines.append(bottleneck)
        lines.append("")

    # Footer: instructions to Stage-2 consumer (important framing)
    lines.append("---")
    lines.append(
        f"*This recipe was distilled from a CUDA solution by `{winner_model}` that "
        f"achieves **{winner_speedup:.1f}× end-to-end speedup** on the medium-scale input. "
        f"Follow the techniques — do not copy literal code.*"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", default="runs/strategy_transfer/best")
    ap.add_argument("--out-dir", default="runs/strategy_transfer/guidance")
    ap.add_argument("--analyzer", default="gemini-3.1-pro-preview-openrouter",
                    help="Model ID (from models.yaml) to use as analyzer")
    ap.add_argument("--min-speedup", type=float, default=2.0,
                    help="Skip tasks whose winner speedup is below this")
    ap.add_argument("--only-tasks", nargs="*", default=None,
                    help="Restrict to these task IDs")
    ap.add_argument("--max-new", type=int, default=2,
                    help="Discovery budget for new patterns (passed to agent analyzer)")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if guidance already exists")
    args = ap.parse_args()

    best_dir = ROOT / args.best_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = best_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no best/manifest.json at {manifest_path}; run extract_best_solutions.py first")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]

    # Load 43-pattern knowledge base
    kb = KnowledgeBase(data_dir=str(KB_DATA_DIR))
    print(f"[kb] loaded {kb.num_patterns()} patterns from {KB_DATA_DIR}")

    # Pick analyzer client
    registry = LLMRegistry()
    client = registry.get_client(args.analyzer)

    selected = []
    for task_id, info in sorted(tasks.items()):
        if args.only_tasks and task_id not in args.only_tasks:
            continue
        if info["winner_speedup"] < args.min_speedup:
            continue
        selected.append((task_id, info))
    print(f"[plan] {len(selected)} tasks eligible (speedup >= {args.min_speedup}×)")

    guidance_manifest = {
        "analyzer_model": args.analyzer,
        "min_speedup": args.min_speedup,
        "kb_num_patterns": kb.num_patterns(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "entries": {},
    }

    for i, (task_id, info) in enumerate(selected, 1):
        out_path = out_dir / f"{task_id}.md"
        if out_path.exists() and not args.force:
            print(f"[skip] {task_id} (guidance exists; --force to rewrite)")
            continue

        cu_path = best_dir / task_id / "solution.cu"
        if not cu_path.exists():
            print(f"[skip] {task_id}: no winner solution.cu")
            continue

        # Step 1: static features
        try:
            auto_features = extract_auto_features(str(cu_path))
        except Exception as e:
            print(f"[skip] {task_id}: static detect failed: {e}")
            continue

        # Step 2: match against the 43-pattern catalog
        with open(cu_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        ptxas_info: dict = {}  # we don't have ptxas info for the winner cache; KB handles missing gracefully
        matched = kb.match_by_features(auto_features, ptxas_info, source_code=source)["matched"]

        # Load task description (for the guidance header)
        task_desc = ""
        try:
            import yaml
            tmpl_p = ROOT / "tasks" / task_id / "prompt_template.yaml"
            if tmpl_p.exists():
                task_desc = (yaml.safe_load(tmpl_p.read_text()).get("task_description") or "").strip()
        except Exception:
            pass

        print(f"[{i}/{len(selected)}] {task_id} "
              f"(winner={info['winner_model']}, {info['winner_speedup']:.1f}×, "
              f"auto-matched={len(matched)}/{kb.num_patterns()})", flush=True)

        # Step 3: unified agent analyzer (single LLM call)
        t0 = time.monotonic()
        try:
            analysis = analyze_sample(
                task_id=task_id,
                source_path=str(cu_path),
                auto_features=auto_features,
                ptxas_info=[],
                benchmark={"speedup_e2e": info["winner_speedup"]},
                knowledge_base=kb,
                auto_matched=matched,
                llm_client=client,
                agent_model_id=args.analyzer,
                max_new=args.max_new,
            )
        except Exception as e:
            print(f"  ✗ analyzer call failed: {e}")
            continue
        dt = time.monotonic() - t0

        if analysis.get("parse_error"):
            print(f"  ! analyzer JSON parse error, skipping")
            continue

        # Step 4: render guidance.md
        md = render_guidance_md(
            task_id=task_id,
            task_desc=task_desc,
            winner_model=info["winner_model"],
            winner_speedup=info["winner_speedup"],
            kb_num_patterns=kb.num_patterns(),
            analysis=analysis,
        )
        out_path.write_text(md)

        # Record manifest entry (incl. raw analysis for downstream inspection)
        guidance_manifest["entries"][task_id] = {
            "winner_model": info["winner_model"],
            "winner_speedup": info["winner_speedup"],
            "guidance_path": str(out_path.relative_to(ROOT)),
            "auto_matched_pattern_ids": [m["pattern_id"] for m in matched],
            "auto_matched_count": len(matched),
            "manual_verified_count": sum(
                1 for p in analysis.get("pattern_summaries", [])
                if p.get("source") == "manual_check"
            ),
            "new_candidate_count": len(analysis.get("new_candidates", [])),
            "strategy_summary": analysis.get("strategy_summary", ""),
            "bottleneck_analysis": analysis.get("bottleneck_analysis", ""),
            "latency_s": round(dt, 2),
        }
        # Also dump the raw analyzer JSON for later paper/figure use
        (out_dir / f"{task_id}.raw.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False)
        )

        print(f"  ✓ wrote {out_path.name} "
              f"({len(analysis.get('pattern_summaries',[]))} patterns, "
              f"{len(analysis.get('new_candidates',[]))} new, "
              f"{dt:.1f}s)")

    # Save manifest (merge with existing if present)
    gman_path = out_dir / "guidance_manifest.json"
    if gman_path.exists() and not args.force:
        old = json.loads(gman_path.read_text())
        old.setdefault("entries", {}).update(guidance_manifest["entries"])
        old["analyzer_model"] = args.analyzer
        old["min_speedup"] = args.min_speedup
        old["kb_num_patterns"] = guidance_manifest["kb_num_patterns"]
        guidance_manifest = old
    gman_path.write_text(json.dumps(guidance_manifest, indent=2))
    print(f"\n[done] {len(guidance_manifest['entries'])} guidance files -> {out_dir}")


if __name__ == "__main__":
    main()
