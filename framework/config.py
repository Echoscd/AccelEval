"""
config.py - Unified configuration loading and management

Loads config.yaml and merges CLI arguments, providing a frozen config object
that all modules can use consistently.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .task import ORBENCH_ROOT


@dataclass(frozen=True)
class LLMConfig:
    """LLM API configuration"""
    model: str = "claude-sonnet-4-20250514"
    api_base: str = "https://api.anthropic.com/v1/messages"
    max_tokens: int = 8192
    num_samples: int = 3
    temperature: float = 0.7


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings"""
    num_gpu_devices: int = 1
    num_cpu_workers: int = 8
    timeout: int = 180
    warmup: int = 3
    num_trials: int = 10


@dataclass(frozen=True)
class GPUConfig:
    """GPU architecture configuration"""
    arch: str = "sm_89"


@dataclass(frozen=True)
class PathsConfig:
    """Path configuration"""
    tasks_dir: str = "tasks"
    runs_dir: str = "runs"
    cache_dir: str = "cache"


@dataclass(frozen=True)
class ProfilingConfig:
    """Profiling settings"""
    nsys_enabled: bool = True
    ncu_enabled: bool = False
    nsys_timeout: int = 60


@dataclass(frozen=True)
class Config:
    """Main configuration object - frozen to prevent accidental modification"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary"""
        return cls(
            llm=LLMConfig(**data.get("llm", {})),
            eval=EvalConfig(**data.get("eval", {})),
            gpu=GPUConfig(**data.get("gpu", {})),
            paths=PathsConfig(**data.get("paths", {})),
            profiling=ProfilingConfig(**data.get("profiling", {})),
        )

    def to_dict(self) -> dict:
        """Convert Config to dictionary"""
        return {
            "llm": {
                "model": self.llm.model,
                "api_base": self.llm.api_base,
                "max_tokens": self.llm.max_tokens,
                "num_samples": self.llm.num_samples,
                "temperature": self.llm.temperature,
            },
            "eval": {
                "num_gpu_devices": self.eval.num_gpu_devices,
                "num_cpu_workers": self.eval.num_cpu_workers,
                "timeout": self.eval.timeout,
                "warmup": self.eval.warmup,
                "num_trials": self.eval.num_trials,
            },
            "gpu": {
                "arch": self.gpu.arch,
            },
            "paths": {
                "tasks_dir": self.paths.tasks_dir,
                "runs_dir": self.paths.runs_dir,
                "cache_dir": self.paths.cache_dir,
            },
            "profiling": {
                "nsys_enabled": self.profiling.nsys_enabled,
                "ncu_enabled": self.profiling.ncu_enabled,
                "nsys_timeout": self.profiling.nsys_timeout,
            },
        }


def load_config_file(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml (default: ORBENCH_ROOT/config.yaml)
    
    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = os.path.join(ORBENCH_ROOT, "config.yaml")
    
    if not os.path.exists(config_path):
        # Return empty dict if config file doesn't exist (use defaults)
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return data


def merge_cli_args(config: Config, cli_args: dict) -> Config:
    """
    Merge CLI arguments into config, overriding YAML defaults.
    
    Args:
        config: Base configuration from YAML
        cli_args: Dictionary of CLI arguments to override (None values are ignored)
    
    Returns:
        New frozen Config object with merged values
    """
    # Build override dictionaries for each section
    llm_overrides = {}
    eval_overrides = {}
    gpu_overrides = {}
    profiling_overrides = {}
    
    # Map CLI argument names to config sections
    # LLM args
    if "model" in cli_args and cli_args["model"] is not None:
        llm_overrides["model"] = cli_args["model"]
    if "api_base" in cli_args and cli_args["api_base"] is not None:
        llm_overrides["api_base"] = cli_args["api_base"]
    if "samples" in cli_args and cli_args["samples"] is not None:
        llm_overrides["num_samples"] = cli_args["samples"]
    
    # Eval args
    if "gpus" in cli_args and cli_args["gpus"] is not None:
        eval_overrides["num_gpu_devices"] = cli_args["gpus"]
    if "timeout" in cli_args and cli_args["timeout"] is not None:
        eval_overrides["timeout"] = cli_args["timeout"]
    
    # GPU args
    if "arch" in cli_args and cli_args["arch"] is not None:
        gpu_overrides["arch"] = cli_args["arch"]
    
    # Profiling args
    if "no_nsys" in cli_args and cli_args["no_nsys"]:
        profiling_overrides["nsys_enabled"] = False
    elif "run_nsys" in cli_args and cli_args["run_nsys"] is not None:
        profiling_overrides["nsys_enabled"] = cli_args["run_nsys"]
    
    # Create new config with overrides (only apply if there are any overrides)
    new_llm = LLMConfig(**{**config.llm.__dict__, **llm_overrides}) if llm_overrides else config.llm
    new_eval = EvalConfig(**{**config.eval.__dict__, **eval_overrides}) if eval_overrides else config.eval
    new_gpu = GPUConfig(**{**config.gpu.__dict__, **gpu_overrides}) if gpu_overrides else config.gpu
    new_profiling = ProfilingConfig(**{**config.profiling.__dict__, **profiling_overrides}) if profiling_overrides else config.profiling
    
    return Config(
        llm=new_llm,
        eval=new_eval,
        gpu=new_gpu,
        paths=config.paths,  # paths typically don't change via CLI
        profiling=new_profiling,
    )


def load_config(config_path: Optional[str] = None, cli_args: Optional[dict] = None) -> Config:
    """
    Load configuration from YAML file and merge CLI arguments.
    
    Args:
        config_path: Path to config.yaml (default: ORBENCH_ROOT/config.yaml)
        cli_args: Dictionary of CLI arguments to override YAML defaults
    
    Returns:
        Frozen Config object ready to use
    """
    # Load YAML config
    yaml_data = load_config_file(config_path)
    
    # Create base config from YAML (with defaults for missing keys)
    config = Config.from_dict(yaml_data)
    
    # Merge CLI args if provided
    if cli_args:
        config = merge_cli_args(config, cli_args)
    
    return config


# Global config instance (lazy-loaded)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    Should be called after load_config() has been called at least once.
    
    Returns:
        Global Config object
    """
    global _global_config
    if _global_config is None:
        # Load default config if not already loaded
        _global_config = load_config()
    return _global_config


def set_config(config: Config):
    """
    Set the global configuration instance.
    Typically called from run.py after loading and merging CLI args.
    
    Args:
        config: Config object to set as global
    """
    global _global_config
    _global_config = config

