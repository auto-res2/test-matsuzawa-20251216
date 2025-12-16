#!/usr/bin/env python
"""
Independent evaluation and visualization script.
Execution: uv run python -m src.evaluate results_dir={path} run_ids='["run-1", "run-2"]'
NOT called from main.py - executes as separate workflow after all training completes.
"""

import os
import sys
import json
import argparse
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate PAVGLS experiment results")
    parser.add_argument("results_dir", type=str, help="Results directory path")
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs")
    return parser.parse_args()


def load_wandb_config() -> Dict[str, str]:
    """Load WandB configuration from config/config.yaml"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return {'entity': 'gengaru617-personal', 'project': '2025-11-19'}
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("wandb", {})
    except:
        return {'entity': 'gengaru617-personal', 'project': '2025-11-19'}


def retrieve_wandb_data(run_id: str, entity: str, project: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Retrieve comprehensive data from WandB for a single run"""
    
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        history = run.history()
        summary = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
        config = dict(run.config)
        
        return history, summary, config
    except Exception as e:
        print(f"Warning: Failed to retrieve data for run {run_id}: {e}")
        return pd.DataFrame(), {}, {}


def export_per_run_metrics(
    run_id: str,
    history: pd.DataFrame,
    summary: Dict[str, Any],
    results_dir: Path
) -> None:
    """Export per-run metrics to JSON and generate per-run figures"""
    
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export comprehensive metrics
    metrics_dict = {
        "run_id": run_id,
        "summary": summary,
        "history_length": len(history),
    }
    
    if not history.empty:
        for col in history.columns:
            if col not in ["_step", "_runtime"]:
                if pd.api.types.is_numeric_dtype(history[col]):
                    metrics_dict[f"timeseries_{col}"] = {
                        "mean": float(history[col].mean()),
                        "std": float(history[col].std()),
                        "min": float(history[col].min()),
                        "max": float(history[col].max()),
                        "final": float(history[col].iloc[-1]),
                    }
    
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Exported metrics: {metrics_file}")
    
    _generate_per_run_figures(run_id, history, summary, run_dir)


def _generate_per_run_figures(
    run_id: str,
    history: pd.DataFrame,
    summary: Dict[str, Any],
    run_dir: Path
) -> None:
    """Generate figures for a single run"""
    
    if history.empty:
        print(f"No history data for {run_id}, skipping figure generation")
        return
    
    # Learning curve
    if "train_loss" in history.columns and "val_loss" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.index, history["train_loss"], label="Training Loss", marker="o", markersize=3)
        ax.plot(history.index, history["val_loss"], label="Validation Loss", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Learning Curve - {run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = run_dir / f"{run_id}_learning_curve.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Accuracy curve
    if "train_acc" in history.columns and "val_acc" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.index, history["train_acc"], label="Training Accuracy", marker="o", markersize=3)
        ax.plot(history.index, history["val_acc"], label="Validation Accuracy", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy Curve - {run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if "val_acc" in history.columns and len(history) > 0:
            final_val_acc = history["val_acc"].iloc[-1]
            ax.axhline(y=final_val_acc * 0.95, color="r", linestyle="--", alpha=0.5, label="95% Target")
        plt.tight_layout()
        fig_path = run_dir / f"{run_id}_accuracy_curve.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Variance trajectory
    if "avg_variance" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.index, history["avg_variance"], label="Gradient Variance", marker="o", markersize=3, color="purple")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance")
        ax.set_title(f"Gradient Variance Trajectory - {run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = run_dir / f"{run_id}_variance_trajectory.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Generated figure: {fig_path}")
        plt.close()
    
    # Learning rate adaptation
    if "avg_lr" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(history.index, history["avg_lr"], label="Effective Learning Rate", marker="o", markersize=3, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate (log scale)")
        ax.set_title(f"Learning Rate Adaptation - {run_id}")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        fig_path = run_dir / f"{run_id}_learning_rate.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Generated figure: {fig_path}")
        plt.close()


def aggregate_metrics(
    run_ids: List[str],
    histories_dict: Dict[str, pd.DataFrame],
    summaries_dict: Dict[str, Dict[str, Any]],
    results_dir: Path
) -> Dict[str, Any]:
    """Aggregate metrics across all runs"""
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated = {
        "primary_metric": "test_accuracy_final",
        "metrics": {},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }
    
    all_metric_names = set()
    for summary in summaries_dict.values():
        all_metric_names.update(summary.keys())
    
    for metric_name in all_metric_names:
        if metric_name.startswith("_"):
            continue
        
        aggregated["metrics"][metric_name] = {}
        for run_id in run_ids:
            if run_id in summaries_dict and metric_name in summaries_dict[run_id]:
                value = summaries_dict[run_id][metric_name]
                if isinstance(value, (int, float)):
                    aggregated["metrics"][metric_name][run_id] = float(value)
    
    # Find best proposed and baseline methods
    primary_metric = "test_accuracy_final"
    if primary_metric in aggregated["metrics"]:
        metric_values = aggregated["metrics"][primary_metric]
        
        proposed_runs = {k: v for k, v in metric_values.items() if "proposed" in k}
        if proposed_runs:
            best_proposed_id = max(proposed_runs, key=proposed_runs.get)
            aggregated["best_proposed"] = {
                "run_id": best_proposed_id,
                "value": proposed_runs[best_proposed_id],
            }
        
        baseline_runs = {k: v for k, v in metric_values.items() if "comparative" in k or "baseline" in k}
        if baseline_runs:
            best_baseline_id = max(baseline_runs, key=baseline_runs.get)
            aggregated["best_baseline"] = {
                "run_id": best_baseline_id,
                "value": baseline_runs[best_baseline_id],
            }
        
        if aggregated["best_proposed"] and aggregated["best_baseline"]:
            proposed_val = aggregated["best_proposed"]["value"]
            baseline_val = aggregated["best_baseline"]["value"]
            if baseline_val != 0:
                # For test_accuracy_final, higher is better (no sign reversal needed)
                gap = ((proposed_val - baseline_val) / abs(baseline_val)) * 100
                aggregated["gap"] = gap
    
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics: {agg_file}")
    
    return aggregated


def generate_comparison_figures(
    run_ids: List[str],
    summaries_dict: Dict[str, Dict[str, Any]],
    results_dir: Path,
    aggregated: Dict[str, Any]
) -> None:
    """Generate comparison figures across runs"""
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    primary_metric = "test_accuracy_final"
    if primary_metric in aggregated["metrics"]:
        metrics_data = aggregated["metrics"][primary_metric]
        
        if metrics_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            run_names = list(metrics_data.keys())
            accuracies = list(metrics_data.values())
            
            colors = ["green" if "proposed" in r else "blue" for r in run_names]
            bars = ax.bar(range(len(run_names)), accuracies, color=colors, alpha=0.7, edgecolor="black")
            
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax.text(bar.get_x() + bar.get_width() / 2, acc, f"{acc:.4f}", 
                       ha="center", va="bottom", fontsize=9)
            
            ax.set_xlabel("Run ID")
            ax.set_ylabel("Test Accuracy")
            ax.set_title("Test Accuracy Comparison Across Runs")
            ax.set_xticklabels(run_names, rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            fig_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Generated figure: {fig_path}")
            plt.close()
    
    # Gap visualization
    if aggregated["gap"] is not None and aggregated["best_proposed"] and aggregated["best_baseline"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ["Baseline", "Proposed"]
        values = [
            aggregated["best_baseline"]["value"],
            aggregated["best_proposed"]["value"]
        ]
        colors = ["blue", "green"]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", 
                   ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Performance Gap: {aggregated['gap']:.2f}%")
        ax.set_ylim([min(values) * 0.95, max(values) * 1.05])
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig_path = comparison_dir / "comparison_performance_gap.pdf"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Generated figure: {fig_path}")
        plt.close()


def main():
    """Main evaluation entry point"""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    wandb_cfg = load_wandb_config()
    entity = wandb_cfg.get("entity", "gengaru617-personal")
    project = wandb_cfg.get("project", "2025-11-19")
    
    print(f"WandB: entity={entity}, project={project}")
    
    histories_dict = {}
    summaries_dict = {}
    
    print("\n" + "="*80)
    print("STEP 1: Retrieving data from WandB for each run")
    print("="*80 + "\n")
    
    for run_id in run_ids:
        print(f"Retrieving data for {run_id}...", end=" ")
        history, summary, config = retrieve_wandb_data(run_id, entity, project)
        histories_dict[run_id] = history
        summaries_dict[run_id] = summary
        print(f"OK (summary keys: {len(summary)})")
    
    print("\n" + "="*80)
    print("STEP 2: Per-Run Processing")
    print("="*80 + "\n")
    
    for run_id in run_ids:
        print(f"Processing {run_id}...")
        export_per_run_metrics(
            run_id,
            histories_dict[run_id],
            summaries_dict[run_id],
            results_dir
        )
    
    print("\n" + "="*80)
    print("STEP 3: Aggregated Analysis")
    print("="*80 + "\n")
    
    aggregated = aggregate_metrics(run_ids, histories_dict, summaries_dict, results_dir)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']}")
    print(f"Best baseline: {aggregated['best_baseline']}")
    print(f"Gap: {aggregated['gap']:.2f}%" if aggregated['gap'] is not None else "Gap: N/A")
    
    print("\n" + "="*80)
    print("Generating comparison figures...")
    print("="*80 + "\n")
    
    generate_comparison_figures(run_ids, summaries_dict, results_dir, aggregated)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {results_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
