"""
render_template.py - Template rendering for generating .cu files

Renders templates from framework/templates/ with task-specific code snippets
to generate LLM_input.cu and cpu_reference.cu files.
"""

import os
from typing import Optional, Dict
from pathlib import Path

from .task import ORBENCH_ROOT, TASKS_DIR, load_task
from .config import get_config


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def load_template(template_name: str) -> str:
    """
    Load a template file.
    
    Args:
        template_name: Name of template file (e.g., "gpu_template.cu")
    
    Returns:
        Template content as string
    """
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def load_algorithm_code(task_id: str) -> str:
    """
    Load algorithm code from task's algorithm.cu file.
    
    Args:
        task_id: Task identifier
    
    Returns:
        Algorithm code as string, or empty string if file doesn't exist
    """
    task_dir = os.path.join(TASKS_DIR, task_id)
    algorithm_path = os.path.join(task_dir, "algorithm.cu")
    
    if not os.path.exists(algorithm_path):
        return ""
    
    with open(algorithm_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove file header comments (lines starting with // that are not code)
    lines = content.split('\n')
    filtered_lines = []
    skip_header = True
    
    for line in lines:
        stripped = line.strip()
        # Skip initial comment-only lines
        if skip_header:
            if stripped.startswith("//") and "algorithm.cu" in stripped:
                continue
            if stripped.startswith("//") and "This file" in stripped:
                continue
            if stripped.startswith("//") and "The I/O" in stripped:
                continue
            if not stripped or stripped.startswith("//"):
                continue
            skip_header = False
        
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def get_task_specific_replacements(task_id: str) -> Dict[str, str]:
    """
    Get task-specific replacement values.
    
    Args:
        task_id: Task identifier
    
    Returns:
        Dictionary mapping placeholder names to replacement values
    """
    task = load_task(task_id)
    config = get_config()
    
    # Determine warmup and num_trials (task config overrides global config)
    warmup = task.warmup if task.warmup > 0 else config.eval.warmup
    num_trials = task.trials if task.trials > 0 else config.eval.num_trials
    
    replacements = {
        "TASK_ID": task_id,
        "WARMUP": str(warmup),
        "NUM_TRIALS": str(num_trials),
    }
    
    return replacements


def get_bellman_ford_replacements() -> Dict[str, str]:
    """
    Get Bellman-Ford specific replacements.
    This is task-specific logic - other tasks may need different replacements.
    """
    return {
        "INCLUDES": "",  # No extra includes needed
        "DATA_STRUCTURES": """struct CSRGraph {
    int num_nodes, num_edges;
    int* row_offsets;    // size: num_nodes + 1
    int* col_indices;    // size: num_edges
    float* weights;      // size: num_edges
};""",
        "BUILD_GRAPH_CODE": """CSRGraph g;
    g.num_nodes = V;
    g.num_edges = E;
    g.row_offsets = row_offsets;
    g.col_indices = col_indices;
    g.weights = weights;""",
        "CALL_ALGORITHM": "gpu_bellman_ford(&g, source, dist);",
        "CLEANUP_CODE": """free(row_offsets);
    free(col_indices);
    free(weights);""",
    }


def extract_gpu_function(algorithm_code: str) -> str:
    """
    Extract GPU algorithm function from algorithm code.
    Looks for function starting with 'void gpu_' or 'void gpu_<task_name>'.
    """
    # Try to find gpu_ function
    if "void gpu_" in algorithm_code:
        func_start = algorithm_code.find("void gpu_")
        # Find the matching closing brace
        brace_count = 0
        func_end = func_start
        in_function = False
        
        for i in range(func_start, len(algorithm_code)):
            if algorithm_code[i] == '{':
                brace_count += 1
                in_function = True
            elif algorithm_code[i] == '}':
                brace_count -= 1
                if brace_count == 0 and in_function:
                    func_end = i + 1
                    break
        
        if func_end > func_start:
            return algorithm_code[func_start:func_end]
    
    # Fallback: return all algorithm code
    return algorithm_code


def extract_cpu_function(algorithm_code: str) -> str:
    """
    Extract CPU algorithm function from algorithm code.
    Looks for function containing '_cpu' in name.
    """
    # Try to find _cpu function
    if "_cpu" in algorithm_code:
        # Find all function definitions
        lines = algorithm_code.split('\n')
        func_start_idx = None
        
        for i, line in enumerate(lines):
            if "void " in line and "_cpu" in line:
                func_start_idx = i
                break
        
        if func_start_idx is not None:
            # Find the matching closing brace
            brace_count = 0
            func_end_idx = func_start_idx
            in_function = False
            
            for i in range(func_start_idx, len(lines)):
                brace_count += lines[i].count('{') - lines[i].count('}')
                if '{' in lines[i]:
                    in_function = True
                if '}' in lines[i] and in_function:
                    if brace_count == 0:
                        func_end_idx = i + 1
                        break
            
            if func_end_idx > func_start_idx:
                return '\n'.join(lines[func_start_idx:func_end_idx])
    
    # Fallback: return all algorithm code
    return algorithm_code


def render_gpu_template(task_id: str, algorithm_code: str = None) -> str:
    """
    Render GPU template for a task.
    
    Args:
        task_id: Task identifier
        algorithm_code: Optional algorithm code (if None, loads from algorithm.cu)
    
    Returns:
        Rendered template as string
    """
    template = load_template("gpu_template.cu")
    
    if algorithm_code is None:
        algorithm_code = load_algorithm_code(task_id)
    
    # Get replacements
    replacements = get_task_specific_replacements(task_id)
    
    # Task-specific replacements (this could be made more generic)
    if task_id == "bellman_ford":
        task_replacements = get_bellman_ford_replacements()
    else:
        # Default replacements - other tasks can override
        task_replacements = {
            "INCLUDES": "",
            "DATA_STRUCTURES": "",
            "BUILD_GRAPH_CODE": "",
            "CALL_ALGORITHM": "",
            "CLEANUP_CODE": "",
        }
    
    # Merge replacements
    all_replacements = {**replacements, **task_replacements}
    
    # Extract GPU function and any helper code
    gpu_function = extract_gpu_function(algorithm_code)
    # Include any code before the GPU function (kernels, helpers, etc.)
    if "void gpu_" in algorithm_code:
        gpu_func_start = algorithm_code.find("void gpu_")
        helper_code = algorithm_code[:gpu_func_start].strip()
        all_replacements["ALGORITHM_CODE"] = helper_code
        all_replacements["ALGORITHM_FUNCTION"] = gpu_function
    else:
        all_replacements["ALGORITHM_CODE"] = ""
        all_replacements["ALGORITHM_FUNCTION"] = gpu_function
    
    # Apply replacements
    result = template
    for placeholder, value in all_replacements.items():
        result = result.replace("{{" + placeholder + "}}", value)
    
    return result


def render_cpu_template(task_id: str, algorithm_code: str = None) -> str:
    """
    Render CPU template for a task.
    
    Args:
        task_id: Task identifier
        algorithm_code: Optional algorithm code (if None, loads from algorithm.cu)
    
    Returns:
        Rendered template as string
    """
    template = load_template("cpu_template.cu")
    
    if algorithm_code is None:
        algorithm_code = load_algorithm_code(task_id)
    
    # Get replacements
    replacements = get_task_specific_replacements(task_id)
    
    # Task-specific replacements
    if task_id == "bellman_ford":
        task_replacements = get_bellman_ford_replacements()
        # For CPU, use CPU function name
        task_replacements["CALL_ALGORITHM"] = "bellman_ford_cpu(&g, source, dist);"
    else:
        task_replacements = {
            "INCLUDES": "",
            "DATA_STRUCTURES": "",
            "BUILD_GRAPH_CODE": "",
            "CALL_ALGORITHM": "",
            "CLEANUP_CODE": "",
        }
    
    # Merge replacements
    all_replacements = {**replacements, **task_replacements}
    
    # Extract CPU function
    cpu_function = extract_cpu_function(algorithm_code)
    all_replacements["ALGORITHM_CODE"] = ""
    all_replacements["ALGORITHM_FUNCTION"] = cpu_function
    
    # Apply replacements
    result = template
    for placeholder, value in all_replacements.items():
        result = result.replace("{{" + placeholder + "}}", value)
    
    return result


def generate_task_files(task_id: str, output_dir: Optional[str] = None):
    """
    Generate LLM_input.cu and cpu_reference.cu for a task.
    
    Args:
        task_id: Task identifier
        output_dir: Output directory (default: tasks/{task_id}/)
    """
    if output_dir is None:
        output_dir = os.path.join(TASKS_DIR, task_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Render templates
    gpu_code = render_gpu_template(task_id)
    cpu_code = render_cpu_template(task_id)
    
    # Write files
    gpu_path = os.path.join(output_dir, "LLM_input.cu")
    cpu_path = os.path.join(output_dir, "cpu_reference.cu")
    
    with open(gpu_path, "w", encoding="utf-8") as f:
        f.write(gpu_code)
    
    with open(cpu_path, "w", encoding="utf-8") as f:
        f.write(cpu_code)
    
    print(f"Generated {gpu_path}")
    print(f"Generated {cpu_path}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Render task templates")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    
    generate_task_files(args.task, args.output_dir)


if __name__ == "__main__":
    main()

