import optuna
import logging
import pandas as pd
import os

# Disable WandB for tuning to avoid spamming the project
os.environ["WANDB_MODE"] = "disabled"

from viberec.tune_modpo import run_modpo_tuning_generic
from viberec.modpo_finetune import run_modpo_finetune

def ml_1m_param_space(trial):
    """
    Define the search space for ML-1M SASRec MODPO.
    ML-1M is larger, so we adjust ranges slightly for stability/speed.
    """
    return {
        # Larger dataset often benefits from stable LRs
        'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 1e-4]),
        
        # Alpha trade-off
        'alpha': trial.suggest_categorical('alpha', [0.8, 0.9]), 
        
        # Group Size
        'group_size': trial.suggest_categorical('group_size', [8, 16]),
        
        # KL Penalty
        'kl_beta': trial.suggest_categorical('kl_beta', [0.05, 0.1]),
        
        'clip_grad_norm': trial.suggest_categorical('clip_grad_norm', [0.5, 1.0]),
        
        # Fewer epochs for tuning on larger dataset
        'epochs': 10
    }

if __name__ == "__main__":
    run_modpo_tuning_generic(
        finetune_func=run_modpo_finetune,
        config_path="examples/config/ml_1m_sasrec_modpo.yaml",
        base_config_path="examples/config/ml_1m_sasrec.yaml",
        n_trials=20,
        output_file='modpo_ml_1m_optuna_results.csv',
        param_space_func=ml_1m_param_space,
        push_best_to_hub=True
    )
