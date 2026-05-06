#!/usr/bin/env python3
"""Merge pass@1 (single-sample) and pass@2 (2-sample) into pass@3 stats.

For each (model, task, size) compute pass@k speedup as the BEST speedup_e2e
over the k samples that compiled AND passed correctness validation.

Output: per-(model, size) aggregate GM speedup and PAR-GM, both for pass@1 and pass@3.
"""
import json, glob, math, os, sys
from collections import defaultdict

# ---------- Pass@1 source ----------
PASS1_FP = sorted(glob.glob('runs/reports/acceleval_consolidated_*.json'))[-1]

# ---------- Pass@2 run dirs (8 models) ----------
PASS2_DIRS = {
    'Gemini 3.1 Pro':  'runs/gemini-3.1-pro-preview-openrouter_l3_20260503_2312',
    'Claude Opus 4.6': 'runs/claude-opus-4.6-openrouter_l3_20260503_2312',
    'GPT-5.4':         'runs/openai/gpt-5.4_l3_20260503_2312',
    'Qwen 3.6 Plus':   'runs/qwen/qwen3.6-plus_l3_20260503_2312',
    'Kimi K2.5':       'runs/kimi-k2.5-openrouter_l3_20260503_2312',
    'GLM 5.1':         'runs/glm-5.1-openrouter_l3_20260503_2312',
    'DeepSeek V3.2':   'runs/deepseek-v3.2-openrouter_l3_20260503_2312',
    'DeepSeek V4 Pro': 'runs/deepseek-v4-pro-openrouter_l3_20260503_2312',
}
MODELS = list(PASS2_DIRS.keys())
SIZES  = ['small', 'medium', 'large']

def latest_eval(run_dir, size_filter=None):
    """Load all eval_results_*.json files in run_dir and return per-(task,sample,size) -> sp."""
    out = {}
    for fp in sorted(glob.glob(f'{run_dir}/eval_results_*.json')):
        try:
            data = json.load(open(fp))
        except Exception: continue
        for k, v in data.items():
            if not isinstance(v, dict): continue
            if not v.get('correct'): continue
            cd = v.get('correctness_detail') or {}
            if size_filter and size_filter not in cd: continue
            t = v.get('task_id', '')
            sid = v.get('sample_id', 0)
            b = v.get('benchmark') or {}
            sz = b.get('size_name') or (list(cd.keys())[0] if cd else None)
            sp = b.get('speedup_e2e')
            if sz is None or sp is None or sp <= 0: continue
            out[(t, sid, sz)] = max(sp, out.get((t, sid, sz), 0.0))
    return out

def main():
    p1 = json.load(open(PASS1_FP))
    print(f'Pass@1 source: {PASS1_FP}', file=sys.stderr)

    # Build best-of-k speedup per (model, task, size) for k=1 and k=3
    # pass@1: only the existing single sample
    # pass@3: max over (pass1 sample, pass2 sample0, pass2 sample1)
    best_p1 = defaultdict(dict)  # [(m, sz)][task] = sp
    best_p3 = defaultdict(dict)
    n_samples_p3 = defaultdict(lambda: defaultdict(int))  # how many of 3 contributed

    # Pass@1 ingest
    for sz in SIZES:
        for t, recs in p1['records'][sz].items():
            for m in MODELS:
                r = recs.get(m)
                if not r or not r.get('correct'): continue
                sp = (r.get('benchmark') or {}).get('speedup_e2e')
                if sp is None or sp <= 0: continue
                best_p1[(m, sz)][t] = sp
                best_p3[(m, sz)][t] = sp
                n_samples_p3[(m, sz)][t] = 1

    # Pass@2 ingest
    for m, run_dir in PASS2_DIRS.items():
        d = latest_eval(run_dir)
        for (t, sid, sz), sp in d.items():
            cur = best_p3[(m, sz)].get(t, 0.0)
            if sp > cur:
                best_p3[(m, sz)][t] = sp
            n_samples_p3[(m, sz)][t] = n_samples_p3[(m, sz)].get(t, 0) + 1

    # PAR-GM = GM over tasks of (s_{m,t} / s*_t), where s*_t = best across models at this size.
    # pass@1 uses pass@1 ceiling (preserves original paper numbers);
    # pass@3 uses pass@3 ceiling (best across the merged sample set).
    star_p1 = {sz: {} for sz in SIZES}
    star_p3 = {sz: {} for sz in SIZES}
    for sz in SIZES:
        for m in MODELS:
            for t, sp in best_p1[(m, sz)].items():
                star_p1[sz][t] = max(star_p1[sz].get(t, 0.0), sp)
            for t, sp in best_p3[(m, sz)].items():
                star_p3[sz][t] = max(star_p3[sz].get(t, 0.0), sp)

    # Apples-to-apples: pass@3 GM restricted to tasks the model already passed at pass@1.
    # Isolates "does multi-sampling improve speedup on already-solvable tasks?" from
    # the dilution effect of newly-covered hard tasks.
    print(f'\n{"Model":<22} {"Size":<7} {"GM_p1":>9} {"GM_p3*":>9} {"LIFT":>7} {"PAR_p1":>8} {"PAR_p3*":>8} {"n":>4}')
    print('  (* GM_p3 / PAR_p3 restricted to tasks_p1; LIFT = GM_p3 / GM_p1 paired)')
    print('-'*90)
    summary = {}
    for m in MODELS:
        summary[m] = {}
        for sz in SIZES:
            tasks_p1 = best_p1[(m, sz)]
            # Restrict pass@3 to the same task set
            tasks_p3_paired = {t: best_p3[(m, sz)][t] for t in tasks_p1 if t in best_p3[(m, sz)]}
            sps_p1 = list(tasks_p1.values())
            sps_p3 = list(tasks_p3_paired.values())
            gm_p1 = math.exp(sum(math.log(s) for s in sps_p1)/len(sps_p1)) if sps_p1 else 0
            gm_p3 = math.exp(sum(math.log(s) for s in sps_p3)/len(sps_p3)) if sps_p3 else 0
            lift = (gm_p3 / gm_p1) if gm_p1 > 0 else 0
            par_p1 = math.exp(sum(math.log(tasks_p1[t]/star_p1[sz][t]) for t in tasks_p1 if star_p1[sz].get(t,0)>0)/len(tasks_p1)) if tasks_p1 else 0
            par_p3 = math.exp(sum(math.log(tasks_p3_paired[t]/star_p1[sz][t]) for t in tasks_p3_paired if star_p1[sz].get(t,0)>0)/len(tasks_p3_paired)) if tasks_p3_paired else 0
            print(f'{m:<22} {sz:<7} {gm_p1:>8.2f}x {gm_p3:>8.2f}x {lift:>6.2f}x {par_p1:>8.2f} {par_p3:>8.2f} {len(sps_p1):>4}')
            summary[m][sz] = dict(gm_p1=gm_p1, gm_p3=gm_p3, lift=lift,
                                  par_p1=par_p1, par_p3=par_p3, n=len(sps_p1))
        print()

    # Compute Pass count (small-scale pass count)
    print('Pass-count at small (correct OR has speedup):')
    for m in MODELS:
        n_p1 = len(best_p1[(m,'small')])
        n_p3 = len(best_p3[(m,'small')])
        print(f'  {m:<22}  pass@1={n_p1}/42  pass@3={n_p3}/42')

    # Save JSON for downstream use
    out_fp = 'runs/reports/pass3_summary.json'
    with open(out_fp, 'w') as f:
        json.dump({'summary': summary, 'pass1_source': PASS1_FP,
                   'star_p1': {sz: dict(star_p1[sz]) for sz in SIZES},
                   'star_p3': {sz: dict(star_p3[sz]) for sz in SIZES}},
                  f, indent=2)
    print(f'\nSaved: {out_fp}', file=sys.stderr)

if __name__ == '__main__':
    main()
