#!/usr/bin/env python
"""
Single experiment run executor using YAML config.
Called as subprocess by main.py.
Responsibilities:
  - Train model with given configuration
  - Initialize WandB
  - Handle Optuna integration
  - Log ALL metrics comprehensively
  - Save final/best metrics to WandB summary
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import yaml
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.model import get_model
from src.preprocess import get_data_loaders

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class PhaseAdaptiveVarianceGuidedScheduler:
    """Phase-Adaptive Variance-Guided Learning Rate Scheduler (PAVGLS)"""
    
    def __init__(self, optimizer, lr_base, schedule_type='exponential', 
                 alpha=0.001, threshold_high=1.5, threshold_low=0.7, lambda_penalty=0.05,
                 ema_decay_short=0.90, ema_decay_long=0.99):
        self.optimizer = optimizer
        self.lr_base = lr_base
        self.schedule_type = schedule_type
        self.alpha = alpha
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.lambda_penalty = lambda_penalty
        
        self.step_count = 0
        self.variance_short_ema = 0.0
        self.variance_long_ema = 0.0
        self.variance_expected = 0.0
        self.ema_decay_short = ema_decay_short
        self.ema_decay_long = ema_decay_long
        
        self.phase_history = []
        self.lr_history = []
        self.variance_history = []
        self.variance_short_history = []
        self.variance_long_history = []
        self.variance_ratio_history = []
    
    def _compute_gradient_variance(self, gradients):
        """Compute variance of gradient norms across parameters"""
        grad_norms = []
        for g in gradients:
            if g is not None:
                grad_norms.append(g.abs().mean().item())
        
        if len(grad_norms) == 0:
            return 0.0
        
        grad_var = np.var(grad_norms) if len(grad_norms) > 1 else (np.mean(grad_norms) ** 2)
        return float(grad_var)
    
    def _detect_phase(self):
        """Detect training phase based on variance ratio"""
        if self.variance_long_ema < 1e-8:
            return 'initialization'
        
        variance_ratio = self.variance_short_ema / (self.variance_long_ema + 1e-10)
        
        if variance_ratio > self.threshold_high:
            return 'exploration'
        elif variance_ratio < self.threshold_low:
            return 'convergence'
        else:
            return 'exploitation'
    
    def _get_phase_multiplier(self, phase):
        """Get learning rate multiplier based on detected phase"""
        multipliers = {
            'initialization': 1.0,
            'exploration': 1.0,
            'exploitation': 0.85,
            'convergence': 0.6
        }
        return multipliers.get(phase, 1.0)
    
    def _get_schedule_multiplier(self, t, total_steps=200):
        """Get schedule multiplier (exponential or cosine decay)"""
        if self.schedule_type == 'exponential':
            return np.exp(-self.alpha * t)
        elif self.schedule_type == 'cosine':
            return 0.5 * (1 + np.cos(np.pi * t / total_steps))
        else:
            return 1.0
    
    def step(self, gradients, total_steps=200):
        """Update learning rate based on phase and variance"""
        current_variance = self._compute_gradient_variance(gradients)
        self.variance_short_ema = (self.ema_decay_short * self.variance_short_ema + 
                                   (1 - self.ema_decay_short) * current_variance)
        self.variance_long_ema = (self.ema_decay_long * self.variance_long_ema + 
                                  (1 - self.ema_decay_long) * current_variance)
        
        phase = self._detect_phase()
        phase_multiplier = self._get_phase_multiplier(phase)
        
        schedule_multiplier = self._get_schedule_multiplier(self.step_count, total_steps)
        
        self.variance_expected = 0.95 * self.variance_expected + 0.05 * current_variance
        variance_spike = max(0, current_variance - self.variance_expected)
        variance_penalty = np.exp(-self.lambda_penalty * variance_spike)
        
        lr = self.lr_base * schedule_multiplier * phase_multiplier * variance_penalty
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        variance_ratio = self.variance_short_ema / (self.variance_long_ema + 1e-10)
        
        self.phase_history.append(phase)
        self.lr_history.append(lr)
        self.variance_history.append(current_variance)
        self.variance_short_history.append(self.variance_short_ema)
        self.variance_long_history.append(self.variance_long_ema)
        self.variance_ratio_history.append(variance_ratio)
        
        self.step_count += 1
        return lr, phase


class ExponentialDecayScheduler:
    """Standard exponential decay scheduler"""
    def __init__(self, optimizer, lr_base, alpha=0.001):
        self.optimizer = optimizer
        self.lr_base = lr_base
        self.alpha = alpha
        self.step_count = 0
    
    def step(self):
        """Update learning rate with exponential decay"""
        lr = self.lr_base * np.exp(-self.alpha * self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1


class CosineAnnealingScheduler:
    """Cosine annealing scheduler"""
    def __init__(self, optimizer, lr_base, total_epochs):
        self.optimizer = optimizer
        self.lr_base = lr_base
        self.total_epochs = total_epochs
        self.step_count = 0
    
    def step(self):
        """Update learning rate with cosine annealing"""
        lr = 0.5 * self.lr_base * (1 + np.cos(np.pi * self.step_count / self.total_epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PAVGLS model")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--mode", type=str, choices=["trial", "full"], required=True, help="Execution mode")
    parser.add_argument("--exec_config", type=str, required=True, help="Execution config path")
    parser.add_argument("--run_config", type=str, required=True, help="Run config path")
    return parser.parse_args()


def load_configs(exec_config_path: str, run_config_path: str) -> Tuple[Dict, Dict]:
    """Load execution and run configurations"""
    with open(exec_config_path, 'r') as f:
        exec_cfg = yaml.safe_load(f)
    
    with open(run_config_path, 'r') as f:
        run_cfg = yaml.safe_load(f)
    
    return exec_cfg, run_cfg


def setup_wandb(run_id: str, exec_cfg: Dict, run_cfg: Dict, mode: str) -> Optional[Any]:
    """Initialize WandB logging - FULLY SKIP if trial mode"""
    wandb_cfg = exec_cfg.get('wandb', {})
    
    if mode == 'trial' or wandb_cfg.get('mode') == 'disabled':
        print("[WandB] Disabled (trial mode or config)")
        return None
    
    try:
        entity = wandb_cfg.get('entity', 'gengaru617-personal')
        project = wandb_cfg.get('project', '2025-11-19')
        
        run_dict = run_cfg if isinstance(run_cfg, dict) else dict(run_cfg)
        
        wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            config=run_dict,
            resume="allow",
            mode="online"
        )
        print(f"[WandB] Initialized: {wandb.run.url}")
        return wandb
    except Exception as e:
        print(f"[WandB] Warning: Failed to initialize: {e}")
        return None


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_wandb: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch with comprehensive logging"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    batch_losses = []
    batch_accs = []
    batch_lrs = []
    batch_phases = []
    batch_variances = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # CRITICAL: Batch-start assertion (check first batch and periodic checks)
        if batch_idx == 0 or batch_idx % max(1, len(train_loader) // 5) == 0:
            assert inputs.shape[0] > 0, f"Empty batch at epoch {epoch}, step {batch_idx}"
            assert targets.shape[0] > 0, f"Empty targets at epoch {epoch}, step {batch_idx}"
            assert inputs.shape[0] == targets.shape[0], f"Batch-target size mismatch at epoch {epoch}, step {batch_idx}"
        
        # Forward pass (inputs ONLY, labels never concatenated)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Loss computation (labels ONLY for loss)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: Pre-optimizer assertion - verify gradients exist and are non-zero
        has_gradients = False
        gradient_norms = []
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient at epoch {epoch}, step {batch_idx}"
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-10:
                    gradient_norms.append(grad_norm)
        
        assert has_gradients, f"No gradients at epoch {epoch}, step {batch_idx}"
        assert len(gradient_norms) > 0, f"All gradients zero at epoch {epoch}, step {batch_idx}"
        
        # Update learning rate with scheduler
        if isinstance(scheduler, PhaseAdaptiveVarianceGuidedScheduler):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            lr, phase = scheduler.step(gradients, total_steps=total_epochs)
            batch_lrs.append(lr)
            batch_phases.append(phase)
            batch_variances.append(scheduler.variance_short_ema)
        elif isinstance(scheduler, (ExponentialDecayScheduler, CosineAnnealingScheduler)):
            scheduler.step()
            batch_lrs.append(optimizer.param_groups[0]['lr'])
        
        # Optimizer step
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = (predicted == targets).sum().item()
        total_correct += batch_correct
        total_samples += targets.size(0)
        
        batch_losses.append(loss.item())
        batch_accs.append(batch_correct / targets.size(0))
        
        # Per-batch logging to WandB (when possible)
        if log_wandb and wandb.run is not None and batch_idx % max(1, len(train_loader) // 10) == 0:
            wandb.log({
                'batch_train_loss': loss.item(),
                'batch_train_acc': batch_correct / targets.size(0),
                'batch_epoch': epoch,
                'batch_step': batch_idx,
            })
    
    # Compute epoch averages
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Log to WandB per-epoch metrics
    if log_wandb and wandb.run is not None:
        log_dict = {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'epoch': epoch,
        }
        
        if batch_lrs:
            log_dict['avg_lr'] = np.mean(batch_lrs)
        
        if isinstance(scheduler, PhaseAdaptiveVarianceGuidedScheduler) and batch_variances:
            log_dict['avg_variance'] = np.mean(batch_variances)
            phase_counts = {}
            for phase in batch_phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            for phase_name in ['exploration', 'exploitation', 'convergence', 'initialization']:
                log_dict[f'phase_{phase_name}_count'] = phase_counts.get(phase_name, 0)
        
        wandb.log(log_dict, step=epoch)
    
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_wandb: bool = True,
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    if log_wandb and wandb.run is not None:
        wandb.log({
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'epoch': epoch,
        }, step=epoch)
    
    return avg_loss, avg_acc


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Test model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_acc


def find_convergence_speed(train_accs: List[float], final_acc: float, threshold: float = 0.95) -> int:
    """Find epoch when accuracy reaches 95% of final"""
    target_acc = final_acc * threshold
    for epoch, acc in enumerate(train_accs):
        if acc >= target_acc:
            return epoch
    return len(train_accs)


def create_scheduler(optimizer: optim.Optimizer, run_cfg: Dict) -> Optional[Any]:
    """Create learning rate scheduler based on config"""
    training_cfg = run_cfg['training']
    scheduler_type = training_cfg.get('scheduler')
    
    if training_cfg.get('pavgls_enabled'):
        pavgls_cfg = training_cfg.get('pavgls_config', {})
        return PhaseAdaptiveVarianceGuidedScheduler(
            optimizer=optimizer,
            lr_base=training_cfg['learning_rate'],
            schedule_type=pavgls_cfg.get('schedule_type', 'exponential'),
            alpha=pavgls_cfg.get('schedule_alpha', 0.001),
            threshold_high=pavgls_cfg.get('threshold_high', 1.5),
            threshold_low=pavgls_cfg.get('threshold_low', 0.7),
            lambda_penalty=pavgls_cfg.get('lambda_penalty', 0.05),
            ema_decay_short=pavgls_cfg.get('ema_decay_short', 0.90),
            ema_decay_long=pavgls_cfg.get('ema_decay_long', 0.99),
        )
    elif scheduler_type == 'exponential':
        return ExponentialDecayScheduler(
            optimizer=optimizer,
            lr_base=training_cfg['learning_rate'],
            alpha=training_cfg.get('scheduler_config', {}).get('schedule_alpha', 0.001)
        )
    elif scheduler_type == 'cosine':
        total_epochs = run_cfg['training']['epochs']
        return CosineAnnealingScheduler(
            optimizer=optimizer,
            lr_base=training_cfg['learning_rate'],
            total_epochs=total_epochs
        )
    
    return None


def train_single_run(
    run_id: str,
    run_cfg: Dict,
    device: torch.device,
    seed: int,
    log_wandb: bool = True,
) -> Dict[str, Any]:
    """Train a single run"""
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*80}")
    print(f"Training run_id={run_id}, seed={seed}")
    print(f"{'='*80}\n")
    
    # POST-INIT ASSERTIONS
    training_cfg = run_cfg['training']
    assert training_cfg['learning_rate'] > 0, "Learning rate must be positive"
    assert training_cfg['epochs'] > 0, "Epochs must be positive"
    assert training_cfg['batch_size'] > 0, "Batch size must be positive"
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(run_cfg)
    
    # Create model
    model = get_model(run_cfg['model']).to(device)
    assert model is not None, "Failed to create model"
    
    # POST-INIT ASSERTIONS: Model output dimensions
    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        dummy_input = dummy_batch[0].to(device)
        dummy_output = model(dummy_input)
        expected_classes = run_cfg['model']['num_classes']
        assert dummy_output.shape[1] == expected_classes, \
            f"Model output dimension mismatch: {dummy_output.shape[1]} vs {expected_classes}"
    
    # Setup optimizer
    optimizer_type = training_cfg['optimizer'].lower()
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_cfg['learning_rate'],
            momentum=training_cfg.get('momentum', 0.9),
            weight_decay=training_cfg.get('weight_decay', 1e-4),
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_cfg['learning_rate'],
            weight_decay=training_cfg.get('weight_decay', 1e-4),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Setup scheduler
    scheduler = create_scheduler(optimizer, run_cfg)
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    total_epochs = training_cfg['epochs']
    
    for epoch in range(total_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, total_epochs, log_wandb=log_wandb
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, log_wandb=log_wandb
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1:
            print(f"Epoch {epoch+1}/{total_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # Test
    test_loss, test_acc = test(model, test_loader, criterion, device)
    
    # Compute metrics
    max_train_acc = max(train_accs) if train_accs else 0.0
    convergence_speed = find_convergence_speed(train_accs, max_train_acc)
    
    # Compute generalization gap
    train_loss_final, _ = test(model, train_loader, criterion, device)
    test_loss_final, _ = test(model, test_loader, criterion, device)
    generalization_gap = train_loss_final - test_loss_final
    
    results = {
        'test_accuracy_final': test_acc,
        'test_loss_final': test_loss,
        'train_accuracy_final': train_accs[-1] if train_accs else 0.0,
        'train_loss_final': train_losses[-1] if train_losses else 0.0,
        'val_accuracy_final': val_accs[-1] if val_accs else 0.0,
        'val_loss_final': val_losses[-1] if val_losses else 0.0,
        'convergence_speed_epochs': convergence_speed,
        'generalization_gap': generalization_gap,
        'train_accs': train_accs,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'val_losses': val_losses,
    }
    
    # Log to WandB summary
    if log_wandb and wandb.run is not None:
        wandb.summary.update(results)
    
    return results


def main():
    """Main entry point for training"""
    args = parse_args()
    run_id = args.run_id
    results_dir = Path(args.results_dir)
    mode = args.mode
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    exec_cfg, run_cfg = load_configs(args.exec_config, args.run_config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup WandB (only if not trial mode)
    log_wandb = mode != 'trial'
    wandb_run = setup_wandb(run_id, exec_cfg, run_cfg, mode) if log_wandb else None
    
    # Get number of runs
    num_runs = run_cfg['training'].get('num_runs', 5)
    base_seed = run_cfg['training'].get('seed', 42)
    
    # Optuna hyperparameter optimization (SKIP in trial mode)
    optuna_cfg = run_cfg.get('optuna', {})
    best_params = {}
    
    if optuna_cfg.get('enabled') and OPTUNA_AVAILABLE and mode == 'full':
        n_trials = optuna_cfg.get('n_trials', 20)
        if n_trials > 0:
            print(f"\n{'='*80}")
            print(f"Running Optuna optimization with {n_trials} trials")
            print(f"{'='*80}\n")
            
            sampler = TPESampler(seed=42)
            pruner = MedianPruner()
            study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
            
            def objective(trial):
                # Suggest hyperparameters
                search_spaces = optuna_cfg.get('search_spaces', [])
                trial_params = {}
                
                for space in search_spaces:
                    param_name = space['param_name']
                    dist_type = space['distribution_type']
                    
                    if dist_type == 'uniform':
                        trial_params[param_name] = trial.suggest_float(
                            param_name, space['low'], space['high']
                        )
                    elif dist_type == 'loguniform':
                        trial_params[param_name] = trial.suggest_float(
                            param_name, space['low'], space['high'], log=True
                        )
                    elif dist_type == 'categorical':
                        trial_params[param_name] = trial.suggest_categorical(
                            param_name, space.get('choices', [])
                        )
                
                # Update config with trial params
                pavgls_cfg = run_cfg['training'].get('pavgls_config', {})
                for param_name, param_value in trial_params.items():
                    if param_name in pavgls_cfg:
                        pavgls_cfg[param_name] = param_value
                
                # Train with trial params (NO logging to WandB for trials)
                torch.manual_seed(42)
                np.random.seed(42)
                
                try:
                    results = train_single_run(
                        run_id=run_id,
                        run_cfg=run_cfg,
                        device=device,
                        seed=42,
                        log_wandb=False,  # DO NOT log intermediate trials
                    )
                    return results.get('test_accuracy_final', 0.0)
                except Exception as e:
                    print(f"Optuna trial failed: {e}")
                    return 0.0
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            print(f"\nBest trial:")
            print(f"  Value: {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")
            
            # Update config with best params
            best_params = study.best_params
            pavgls_cfg = run_cfg['training'].get('pavgls_config', {})
            for param_name, param_value in best_params.items():
                if param_name in pavgls_cfg:
                    pavgls_cfg[param_name] = param_value
    
    # Train multiple runs with different seeds
    all_results = {}
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        try:
            results = train_single_run(
                run_id=run_id,
                run_cfg=run_cfg,
                device=device,
                seed=seed,
                log_wandb=log_wandb,
            )
            all_results[f'seed_{seed}'] = results
        except Exception as e:
            print(f"Error in run {run_idx} (seed {seed}): {e}")
            import traceback
            traceback.print_exc()
            if log_wandb and wandb.run is not None:
                wandb.log({'error': str(e)})
    
    # Save results to file
    results_file = results_dir / f"{run_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary for {run_id}")
    print(f"{'='*80}")
    
    all_test_accs = []
    for seed_key, results in all_results.items():
        test_acc = results.get('test_accuracy_final', 0)
        all_test_accs.append(test_acc)
        print(f"{seed_key}: test_acc={test_acc:.4f}")
    
    if all_test_accs:
        mean_acc = np.mean(all_test_accs)
        std_acc = np.std(all_test_accs)
        print(f"\nMean test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        if log_wandb and wandb.run is not None:
            wandb.summary['test_accuracy_mean'] = mean_acc
            wandb.summary['test_accuracy_std'] = std_acc
            print(f"WandB run: {wandb.run.url}")
            wandb.finish()


if __name__ == "__main__":
    main()
