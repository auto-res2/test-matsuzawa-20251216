#!/usr/bin/env python
"""
Main orchestrator for PAVGLS experiment runs via Hydra.
Receives run_id via CLI, launches train.py as subprocess, manages logs.
Usage: uv run python -u -m src.main run={run_id} results_dir={path} mode=full
"""

import os
import sys
import subprocess
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import yaml

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main orchestrator for PAVGLS experiment runs via Hydra"""
    
    # Extract CLI overrides
    run_id = cfg.get("run")
    results_dir = cfg.get("results_dir")
    mode = cfg.get("mode")
    
    # Validate required parameters
    if not run_id:
        raise ValueError("run parameter must be specified via CLI: run=<run_id>")
    if not results_dir:
        raise ValueError("results_dir parameter must be specified via CLI: results_dir=<path>")
    if not mode:
        raise ValueError("mode parameter must be specified via CLI: mode=<trial|full>")
    
    if mode not in ["trial", "full"]:
        raise ValueError(f"mode must be 'trial' or 'full', got {mode}")
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load run configuration from config/runs/{run_id}.yaml
    # Use absolute path to avoid Hydra working directory issues
    repo_root = Path(__file__).parent.parent.absolute()
    run_config_path = repo_root / "config" / "runs" / f"{run_id}.yaml"
    
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    
    with open(run_config_path, 'r') as f:
        run_cfg = yaml.safe_load(f)
    
    # Create execution configuration based on mode
    exec_cfg = {
        "wandb": {
            "entity": "gengaru617-personal",
            "project": "2025-11-19",
            "mode": "online"
        },
        "mode": mode
    }
    
    # Configure based on mode
    if mode == "trial":
        print(f"\n{'='*80}")
        print(f"[TRIAL MODE] Running lightweight validation")
        print(f"{'='*80}\n")
        run_cfg["training"]["epochs"] = 1
        run_cfg["training"]["num_runs"] = 1
        if "optuna" in run_cfg:
            run_cfg["optuna"]["n_trials"] = 0
            run_cfg["optuna"]["enabled"] = False
        exec_cfg["wandb"]["mode"] = "disabled"
    elif mode == "full":
        print(f"\n{'='*80}")
        print(f"[FULL MODE] Running complete experiment")
        print(f"{'='*80}\n")
        exec_cfg["wandb"]["mode"] = "online"
    
    # Save execution configuration
    exec_config_path = results_dir / f"{run_id}_exec_config.yaml"
    with open(exec_config_path, 'w') as f:
        yaml.dump(exec_cfg, f)
    print(f"Saved execution config: {exec_config_path}")
    
    # Save run configuration
    run_config_output = results_dir / f"{run_id}_run_config.yaml"
    with open(run_config_output, 'w') as f:
        yaml.dump(run_cfg, f)
    print(f"Saved run config: {run_config_output}")
    
    # Execute training as subprocess
    train_script = Path(__file__).parent / "train.py"
    train_cmd = [
        sys.executable, "-u",
        str(train_script),
        f"--run_id={run_id}",
        f"--results_dir={results_dir}",
        f"--mode={mode}",
        f"--exec_config={exec_config_path}",
        f"--run_config={run_config_output}",
    ]
    
    print(f"\n{'='*80}")
    print(f"Launching training subprocess:")
    print(' '.join(train_cmd))
    print(f"{'='*80}\n")
    
    # Set working directory to repo root for proper imports
    result = subprocess.run(train_cmd, cwd=str(repo_root))
    
    if result.returncode != 0:
        print(f"\nERROR: Training failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n{'='*80}")
    print(f"Training completed successfully!")
    print(f"Run ID: {run_id}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
