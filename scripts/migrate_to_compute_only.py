#!/usr/bin/env python3
"""
Migrate init_compute tasks to compute_only (unified interface).

Template-based regeneration of task_io.cu / task_io_cpu.c, plus surgical
edits to cpu_reference.c / prompt_template.yaml / task.json.

task_io.{cu,c} strategy:
  Parse old file to extract:
    - solution_init / solution_compute signatures
    - output allocation size expressions
    - tensor getter calls (get_tensor_float vs _int vs _double)
  Regenerate using a clean template.
"""
import os, re, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"


# ── Param parsing ────────────────────────────────────────────

def parse_params(s: str):
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    s = s.strip()
    if not s or s == "void":
        return []
    parts, depth, cur = [], 0, ""
    for c in s:
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if c == ',' and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += c
    if cur.strip(): parts.append(cur.strip())
    out = []
    for p in parts:
        m = re.match(r'^(.*?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\[\s*\])?\s*$', p)
        if m:
            typ = m.group(1).strip()
            name = m.group(2).strip()
            if m.group(3): typ += "[]"
            out.append((typ, name))
    return out


def find_extern_decl(src: str, fname: str):
    pat = (rf'(?:extern\s*(?:"C")?\s+)?'
           rf'(?:void|int|float|double|unsigned(?:\s+long)*)\s+{fname}\s*\(([^;{{]*?)\)\s*[;{{]')
    m = re.search(pat, src, re.DOTALL)
    return m.group(1).strip() if m else None


def find_matching_brace(src: str, open_idx: int):
    depth = 0
    i = open_idx
    while i < len(src):
        if src[i] == '{': depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(src)


# ── Type → getter mapping ───────────────────────────────────

TYPE_TO_SCALAR_CAST = {
    "int": "(int)",
    "unsigned int": "(unsigned)",
    "unsigned": "(unsigned)",
    "unsigned long": "(unsigned long)",
    "unsigned long long": "(unsigned long long)",
    "long": "(long)",
    "float": "(float)",
    "double": "",
    "size_t": "(size_t)",
}

PTR_BASE_TO_GETTER = {
    "int": "get_tensor_int",
    "float": "get_tensor_float",
    "double": "get_tensor_double",
    "unsigned int": "get_tensor_u32",
    "unsigned": "get_tensor_u32",
    "unsigned long long": "get_tensor_u64",
    "long long": "get_tensor_i64",
    "char": "get_tensor_u8",
}


def is_pointer(typ: str) -> bool:
    return "*" in typ or typ.endswith("[]")


def base_type(typ: str) -> str:
    """Strip const, * and [] to get base type."""
    t = typ.replace("const", "").replace("*", "").replace("[]", "").strip()
    # Normalize whitespace
    t = " ".join(t.split())
    return t


def scalar_getter(typ: str, name: str) -> str:
    base = base_type(typ)
    cast = TYPE_TO_SCALAR_CAST.get(base, f"({base})")
    return f'{cast}get_param(data, "{name}")'


def tensor_getter(typ: str, tensor_name: str) -> str:
    base = base_type(typ)
    getter = PTR_BASE_TO_GETTER.get(base, "get_tensor_float")
    return f'{getter}(data, "{tensor_name}")'


# ── Extract output size expressions from original task_io.cu ──

def extract_output_sizes(orig_src: str, output_params):
    """
    For each output field, find its allocation expression in the original
    task_setup. Looks for patterns like:
        ctx->out = (TYPE*)malloc(SIZE * sizeof(TYPE));
        ctx->out = (TYPE*)calloc(N, sizeof(TYPE));
    Returns dict {name: size_expr_as_string}.
    """
    result = {}
    for typ, name in output_params:
        # match ctx->name = ... with the size inside sizeof(...)
        # Examples:
        #   ctx->y_out = (double*)malloc((size_t)n * sizeof(double));
        #   ctx->labels = (int*)calloc(ctx->N, sizeof(int));
        pat = re.compile(
            rf'ctx\s*->\s*{name}\s*=\s*\([^)]*\)\s*(?:malloc|calloc)\s*\(([^;]+?)\)\s*;',
            re.DOTALL)
        m = pat.search(orig_src)
        if m:
            args = m.group(1).strip()
            # calloc has TWO args separated by top-level comma, malloc has ONE
            depth = 0; parts = [""]
            for c in args:
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                if c == ',' and depth == 0:
                    parts.append("")
                else:
                    parts[-1] += c
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 1:
                # malloc(size_bytes) — strip trailing sizeof part if present
                bytes_expr = parts[0]
                # e.g. "(size_t)n * sizeof(double)"
                m2 = re.match(r'^(.*?)\s*\*\s*sizeof\s*\([^)]*\)\s*$', bytes_expr)
                if m2:
                    result[name] = m2.group(1).strip()
                else:
                    result[name] = bytes_expr
            elif len(parts) >= 2:
                # calloc(n, sizeof(TYPE)) — size is first arg
                result[name] = parts[0]
    return result


# ── Inputs: figure out tensor/scalar mapping ────────────────

def classify_inputs(init_params, orig_src):
    """
    For each init param, find the EXPRESSION originally passed to solution_init
    at that position. The expression is typically a local var defined earlier
    as `TYPE LOCAL = get_tensor_X(data, "tname");` — we resolve that local back
    to its getter call so the new task_io uses the correct tensor name.
    """
    # 1) Extract args passed to solution_init FROM THE ACTUAL CALL inside
    #    task_setup (not the extern declaration, which would also match).
    init_call_args = []
    # Find task_setup body first, then search for solution_init inside it.
    body_src = orig_src
    m_ts = re.search(r'\bvoid\*\s+task_setup\s*\([^)]*\)\s*\{', orig_src)
    if m_ts:
        end = find_matching_brace(orig_src, m_ts.end() - 1)
        body_src = orig_src[m_ts.end():end]
    m = re.search(r'solution_init\s*\(([^;]*?)\)\s*;', body_src, re.DOTALL)
    if m:
        args_src = m.group(1)
        depth, cur = 0, ""
        for c in args_src:
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if c == ',' and depth == 0:
                init_call_args.append(cur.strip()); cur = ""
            else:
                cur += c
        if cur.strip(): init_call_args.append(cur.strip())

    info = []
    for i, (typ, name) in enumerate(init_params):
        entry = {"typ": typ, "name": name, "ptr": is_pointer(typ)}

        # Resolve the arg expression passed for this formal param
        arg_expr = init_call_args[i] if i < len(init_call_args) else name

        # If arg_expr is a bare identifier, chase its definition
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', arg_expr):
            local = arg_expr
            # Look for `... LOCAL = RHS;` in original source
            m2 = re.search(
                rf'\b{re.escape(local)}\s*=\s*([^;]+?)\s*;', orig_src)
            entry["rhs"] = m2.group(1).strip() if m2 else None
        else:
            # arg_expr itself is already an expression
            entry["rhs"] = arg_expr

        # Metadata for fallbacks
        if entry["ptr"]:
            m3 = re.search(r'(get_tensor_\w+)\s*\(\s*data\s*,\s*"([^"]+)"\s*\)',
                           entry.get("rhs") or "")
            entry["getter_fn"] = m3.group(1) if m3 else None
            entry["tensor_name"] = m3.group(2) if m3 else name
        else:
            m3 = re.search(r'get_param\s*\(\s*data\s*,\s*"([^"]+)"\s*\)',
                           entry.get("rhs") or "")
            entry["scalar_key"] = m3.group(1) if m3 else name
        info.append(entry)
    return info


# ── Check if original uses local get_tensor_*_local helpers ──

def extract_local_scalars(orig_src: str, exclude: set[str]) -> list[str]:
    """
    Extract local variable declarations in task_setup that call get_param or
    get_tensor_* (both scalars and tensors). These are helper bindings that
    downstream expressions in the original code may reference.
    Excludes variables whose names are in `exclude` (those become ctx->X).
    """
    m = re.search(r'\bvoid\*\s+task_setup\s*\([^)]*\)\s*\{', orig_src)
    if not m: return []
    end = find_matching_brace(orig_src, m.end() - 1)
    body = orig_src[m.end():end - 1]

    out = []
    # Matches any type spec + name + = RHS containing get_param / get_tensor
    pat = re.compile(
        r'^\s*((?:const\s+)?(?:int|float|double|unsigned\s+int|unsigned\s+long\s+long|'
        r'unsigned|long|size_t|char)\s*\*?\s*'
        r'[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;]*?(?:get_param|get_tensor_\w+)[^;]*?)\s*;',
        re.MULTILINE)
    for match in pat.finditer(body):
        decl_full = match.group(1).strip()
        # Extract variable name (last identifier before =)
        m2 = re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*=', decl_full)
        if not m2: continue
        vname = m2.group(1)
        if vname in exclude:
            continue
        out.append(f'    {decl_full};')
    return out


def extract_derived_fields(orig_src: str, known_inputs: set[str], known_outputs: set[str]):
    """
    Find ctx->FIELD = EXPR assignments in task_setup where FIELD is NOT already
    an input or output we're handling. These are typically derived scalars like
    `ctx->n = rows * cols`. Returns a list of (field_name, type, expr).
    """
    # Isolate task_setup body
    m = re.search(r'\bvoid\*\s+task_setup\s*\([^)]*\)\s*\{', orig_src)
    if not m: return []
    end = find_matching_brace(orig_src, m.end() - 1)
    body = orig_src[m.end():end - 1]

    derived = []
    # Match ctx->FIELD = EXPR; where EXPR is NOT a calloc/malloc/get_param/get_tensor call
    for m2 in re.finditer(
        r'ctx\s*->\s*(\w+)\s*=\s*([^;]+?)\s*;', body):
        field = m2.group(1)
        expr = m2.group(2).strip()
        if field in known_inputs or field in known_outputs:
            continue
        # Skip allocation / getter assignments
        if re.search(r'\b(calloc|malloc|get_param|get_tensor|NULL)\b', expr):
            continue
        derived.append((field, "int", expr))  # assume int by default
    return derived


def extract_local_helpers(orig_src: str) -> str:
    """
    Preserve any `static ... get_tensor_XXX_local(...) { ... }` helpers
    that existed in the original file (needed for typed tensor access).
    """
    out = []
    for m in re.finditer(r'^static[^;\n]*?\bget_tensor_\w+_local\s*\([^)]*\)\s*\{',
                         orig_src, re.MULTILINE):
        start = m.start()
        end = find_matching_brace(orig_src, m.end() - 1)
        out.append(orig_src[start:end])
    return '\n\n'.join(out)


# ── Determine output write format ───────────────────────────

def extract_write_output_body(orig_src: str, ctx_name: str) -> str:
    """
    Extract inner-body (without the ctx declaration line) of task_write_output
    so we preserve output formatting but don't double-declare ctx.
    """
    m = re.search(
        r'void\s+task_write_output\s*\([^)]*\)\s*\{',
        orig_src)
    if not m: return None
    end = find_matching_brace(orig_src, m.end() - 1)
    body = orig_src[m.end():end - 1]
    # Strip any existing ctx declaration line: `CtxName* ctx = (CtxName*)test_data;`
    body = re.sub(
        rf'^\s*{re.escape(ctx_name)}\s*\*\s*ctx\s*=\s*\([^)]*\)\s*test_data\s*;\s*\n',
        '', body, flags=re.MULTILINE)
    return body


# ── Task_io generator ───────────────────────────────────────

def generate_task_io(orig_src, ctx_name, all_inputs, outputs, unified, is_cu):
    ext_prefix = 'extern "C" ' if is_cu else ''
    inc_lines = ['#include "orbench_io.h"',
                 '#include <stdio.h>',
                 '#include <stdlib.h>',
                 '#include <string.h>']
    if is_cu:
        inc_lines.append('#include <cuda_runtime.h>')

    # ── Derived fields + translator (compute early) ──
    known_input_names = {n for _, n in all_inputs}
    known_output_names = {n for _, n in outputs}
    derived = extract_derived_fields(orig_src, known_input_names, known_output_names)
    # Dedupe derived by field name (keep first occurrence)
    seen = set()
    derived_uniq = []
    for d in derived:
        if d[0] in seen:
            continue
        seen.add(d[0])
        derived_uniq.append(d)
    derived = derived_uniq

    ctx_field_names = known_input_names | known_output_names | {d[0] for d in derived}

    def translate_expr(expr: str) -> str:
        """Prefix bare ctx field references with ctx->, skip already-qualified refs
        and skip occurrences inside string literals."""
        # Find string literals and protect them
        literals = []
        def protect(m):
            literals.append(m.group(0))
            return f"__STR_{len(literals)-1}__"
        protected = re.sub(r'"(?:[^"\\]|\\.)*"', protect, expr)

        def _repl(m):
            pos = m.start()
            if pos >= 2 and m.string[pos-2:pos] == '->':
                return m.group(0)
            if pos >= 1 and m.string[pos-1] == '.':
                return m.group(0)
            v = m.group(0)
            if v in ctx_field_names:
                return f'ctx->{v}'
            return v
        translated = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', _repl, protected)

        # Restore literals
        def restore(m):
            idx = int(m.group(1))
            return literals[idx]
        return re.sub(r'__STR_(\d+)__', restore, translated)

    # ── Ctx struct ──
    ctx_fields = []
    for typ, name in all_inputs:
        ctx_fields.append(f'    {typ} {name};')
    for fname, ftype, _expr in derived:
        ctx_fields.append(f'    {ftype} {fname};')
    for typ, name in outputs:
        ctx_fields.append(f'    {typ} {name};')

    # ── Extern decl for solution_compute ──
    sig_lines = ",\n                             ".join(f"{t} {n}" for t, n in unified)
    extern_decl = f'{ext_prefix}void solution_compute({sig_lines});'

    # ── Preserve local static helpers (e.g. get_tensor_int_local) ──
    helpers = extract_local_helpers(orig_src)

    # ── Local scalar decls for RHS references (e.g. eps_x10000) ──
    local_scalar_decls = extract_local_scalars(orig_src, exclude=ctx_field_names)

    # ── Input assignments to ctx fields ──
    input_info = classify_inputs(all_inputs, orig_src)
    setup_lines = list(local_scalar_decls)
    for info in input_info:
        rhs = info.get("rhs")
        if rhs:
            setup_lines.append(f'    ctx->{info["name"]} = {translate_expr(rhs)};')
        elif info["ptr"]:
            getter_fn = info.get("getter_fn") or PTR_BASE_TO_GETTER.get(base_type(info['typ']), 'get_tensor_float')
            setup_lines.append(f'    ctx->{info["name"]} = {getter_fn}(data, "{info["tensor_name"]}");')
        else:
            key = info.get("scalar_key") or info["name"]
            setup_lines.append(f'    ctx->{info["name"]} = {scalar_getter(info["typ"], key)};')

    # ── Validate pointer inputs ──
    ptr_checks = [f'ctx->{info["name"]}' for info in input_info if info["ptr"]]
    if ptr_checks:
        check_expr = " || ".join(f"!{p}" for p in ptr_checks)
        setup_lines.append("")
        setup_lines.append(f'    if ({check_expr}) {{')
        setup_lines.append(f'        fprintf(stderr, "[task_io] Missing tensor data\\n");')
        setup_lines.append(f'        free(ctx);')
        setup_lines.append(f'        return NULL;')
        setup_lines.append(f'    }}')

    # ── Derived-field assignments ──
    for fname, _ftype, fexpr in derived:
        setup_lines.append(f'    ctx->{fname} = {translate_expr(fexpr)};')

    # ── Output buffer allocation ──
    size_map = extract_output_sizes(orig_src, outputs)
    for typ, name in outputs:
        base = base_type(typ)
        size_expr = size_map.get(name, '1')
        size_expr_translated = translate_expr(size_expr)
        size_expr_translated = re.sub(r'^\s*\(size_t\)\s*', '', size_expr_translated).strip()
        setup_lines.append(f'    ctx->{name} = ({base}*)calloc((size_t)({size_expr_translated}), sizeof({base}));')

    # task_run: call solution_compute with all ctx fields
    call_args = ', '.join(f'ctx->{n}' for _, n in unified)

    # task_write_output body (preserve formatting, strip ctx decl)
    write_body = extract_write_output_body(orig_src, ctx_name)
    if write_body is None:
        write_body = '    /* output writing — verify per task */\n'

    # Build the file
    parts = []
    parts.append(f'// task_io{".cu" if is_cu else "_cpu.c"} — unified compute_only interface (auto-migrated)')
    parts.append('')
    parts.append('\n'.join(inc_lines))
    parts.append('')
    parts.append(extern_decl)
    parts.append('')
    parts.append(f'typedef struct {{\n' + '\n'.join(ctx_fields) + f'\n}} {ctx_name};')
    parts.append('')
    if helpers:
        parts.append(helpers)
        parts.append('')
    parts.append(f'{ext_prefix}void* task_setup(const TaskData* data, const char* data_dir) {{')
    parts.append(f'    (void)data_dir;')
    parts.append(f'    {ctx_name}* ctx = ({ctx_name}*)calloc(1, sizeof({ctx_name}));')
    parts.append(f'    if (!ctx) return NULL;')
    parts.append('\n'.join(setup_lines))
    parts.append(f'    return ctx;')
    parts.append(f'}}')
    parts.append('')
    parts.append(f'{ext_prefix}void task_run(void* test_data) {{')
    parts.append(f'    {ctx_name}* ctx = ({ctx_name}*)test_data;')
    parts.append(f'    solution_compute({call_args});')
    parts.append(f'}}')
    parts.append('')
    parts.append(f'{ext_prefix}void task_write_output(void* test_data, const char* output_path) {{')
    parts.append(f'    {ctx_name}* ctx = ({ctx_name}*)test_data;')
    parts.append(write_body.rstrip())
    parts.append(f'}}')
    parts.append('')
    parts.append(f'{ext_prefix}void task_cleanup(void* test_data) {{')
    parts.append(f'    if (!test_data) return;')
    parts.append(f'    {ctx_name}* ctx = ({ctx_name}*)test_data;')
    for _, name in outputs:
        parts.append(f'    free(ctx->{name});')
    parts.append(f'    free(ctx);')
    parts.append(f'}}')

    return '\n'.join(parts) + '\n'


# ── cpu_reference.c rewriter ────────────────────────────────

def update_cpu_reference(path: Path, init_params, compute_params, unified):
    src = path.read_text()
    src = re.sub(r'\bvoid\s+solution_init\s*\(', 'static void _orbench_old_init(', src, count=1)
    src = re.sub(r'\bvoid\s+solution_compute\s*\(', 'static void _orbench_old_compute(', src, count=1)
    # Remove solution_free whole body
    m = re.search(r'\bvoid\s+solution_free\s*\([^)]*\)\s*\{', src)
    if m:
        end = find_matching_brace(src, m.end() - 1)
        src = src[:m.start()] + src[end:]

    unified_sig = ", ".join(f"{t} {n}" for t, n in unified)
    init_call = ", ".join(n for _, n in init_params)
    compute_call = ", ".join(n for _, n in compute_params)
    wrapper = f"""

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute({unified_sig}) {{
    _orbench_old_init({init_call});
    _orbench_old_compute({compute_call});
}}
"""
    src = src.rstrip() + wrapper
    path.write_text(src)


# ── prompt_template.yaml rewriter ───────────────────────────

def update_prompt_yaml(path: Path, unified):
    src = path.read_text()
    sig_lines = ",\n      ".join(f"{t} {n}" for t, n in unified)
    new_block = f"""interface: |
  ```c
  // Single end-to-end call. Do H2D + kernel + D2H and synchronize before returning.
  // Called multiple times (warmup + timed); must be idempotent.
  extern "C" void solution_compute(
      {sig_lines}
  );
  ```
"""
    m = re.search(r'^interface:\s*\|\s*\n(.*?)(?=^[a-zA-Z_][a-zA-Z_0-9]*:\s|\Z)',
                  src, flags=re.DOTALL | re.MULTILINE)
    if m:
        src = src[:m.start()] + new_block + '\n' + src[m.end():]
    else:
        src = src.rstrip() + '\n\n' + new_block
    path.write_text(src)


# ── task.json rewriter ──────────────────────────────────────

def update_task_json(path: Path):
    with open(path) as f: data = json.load(f)
    if data.get("interface_mode") == "compute_only":
        return
    new_data = {}
    inserted = False
    for k, v in data.items():
        new_data[k] = v
        if k == "correctness" and not inserted:
            new_data["interface_mode"] = "compute_only"
            inserted = True
    if not inserted:
        new_data["interface_mode"] = "compute_only"
    with open(path, "w") as f:
        json.dump(new_data, f, indent=2)
        f.write("\n")


# ── Per-task driver ─────────────────────────────────────────

def migrate_task(tid: str, dry_run: bool = False):
    r = {"task": tid, "errors": []}
    td = TASKS_DIR / tid
    if not td.is_dir():
        r["errors"].append("task dir missing"); return r

    io_cu = td / "task_io.cu"
    io_c  = td / "task_io_cpu.c"
    orig_cu = io_cu.read_text()
    orig_c  = io_c.read_text()

    init_sig = find_extern_decl(orig_cu, "solution_init")
    compute_sig = find_extern_decl(orig_cu, "solution_compute")
    if init_sig is None:
        r["errors"].append("no solution_init decl"); return r
    if compute_sig is None:
        r["errors"].append("no solution_compute decl"); return r

    init_params = parse_params(init_sig)
    compute_params = parse_params(compute_sig)
    init_names = {n for _, n in init_params}
    # Split compute params into:
    #   - outputs: non-const pointers (these need allocation)
    #   - extra inputs: scalars or const pointers (additional per-call inputs)
    compute_extra_inputs = []
    compute_outputs = []
    for (t, n) in compute_params:
        if n in init_names:
            continue
        if "*" in t and "const" not in t:
            compute_outputs.append((t, n))
        else:
            compute_extra_inputs.append((t, n))
    compute_params_out = compute_outputs  # kept for backward name
    all_inputs = init_params + compute_extra_inputs
    unified = init_params + compute_extra_inputs + compute_outputs
    r["new_compute"] = unified
    r["_all_inputs"] = all_inputs
    r["_outputs"] = compute_outputs

    # Extract Ctx struct name
    m_ctx = re.search(r'typedef\s+struct\s*\{[^}]*\}\s*(\w+)\s*;', orig_cu, re.DOTALL)
    ctx_name = m_ctx.group(1) if m_ctx else f"{tid.title().replace('_','')}Context"

    if dry_run:
        return r

    try:
        io_cu.write_text(generate_task_io(orig_cu, ctx_name, all_inputs, compute_outputs, unified, is_cu=True))
        io_c.write_text(generate_task_io(orig_c, ctx_name, all_inputs, compute_outputs, unified, is_cu=False))
        # For cpu_reference: call _orbench_old_compute with the ORIGINAL compute_params
        # signature (including params that overlap with init), since that's what the
        # renamed function still takes verbatim.
        update_cpu_reference(td / "cpu_reference.c", init_params, compute_params, unified)
        update_prompt_yaml(td / "prompt_template.yaml", unified)
        update_task_json(td / "task.json")
    except Exception as e:
        import traceback
        r["errors"].append(f"{e}\n{traceback.format_exc()[:500]}")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.tasks:
        targets = args.tasks
    else:
        targets = []
        for td in sorted(TASKS_DIR.iterdir()):
            if not td.is_dir(): continue
            tj = td / "task.json"
            if not tj.exists(): continue
            with open(tj) as f: j = json.load(f)
            if j.get("interface_mode", "init_compute") == "init_compute":
                targets.append(td.name)

    print(f"Migrating {len(targets)}{' (dry-run)' if args.dry_run else ''}")
    ok, fail = [], []
    for tid in targets:
        r = migrate_task(tid, dry_run=args.dry_run)
        if r["errors"]:
            print(f"  ❌ {tid}: {r['errors'][0].splitlines()[0]}")
            fail.append(tid)
        else:
            sig = r.get("new_compute", [])
            preview = ", ".join(f"{t} {n}" for t, n in sig)[:80]
            print(f"  ✅ {tid}: compute({preview}...)")
            ok.append(tid)
    print(f"\n  {len(ok)} OK, {len(fail)} fail")
    if fail: print(f"  Failed: {fail}")


if __name__ == "__main__":
    main()
