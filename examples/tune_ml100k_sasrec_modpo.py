import optuna
import logging
import pandas as pd
import os

# Disable WandB for tuning to avoid spamming the project
os.environ["WANDB_MODE"] = "disabled"

from viberec.tune_modpo import run_modpo_tuning_generic
from examples.finetune_ml100k_sasrec_modpo import run_modpo_finetune

def ml100k_param_space(trial):
    """
    Define the search space for ML-100k SASRec MODPO.
    """
    return {
        # Refined Analysis (targeting Max NDCG > 0.0704)
        # 1e-4 gave highest NDCG. 5e-5 is robust.
        'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 8e-5, 1e-4, 3e-4]),
        
        # Alpha is unused in Lexicographical Logic
        'alpha': 1.0, 
        
        # Group Size 8 is best for pure NDCG. 16 for diversity.
        'group_size': trial.suggest_categorical('group_size', [8, 16]),
        
        # Lower beta (0.05) allows more policy freedom -> Higher NDCG
        'kl_beta': trial.suggest_categorical('kl_beta', [0.05, 0.08, 0.1]),
        
        # Support higher LR stability
        'clip_grad_norm': trial.suggest_categorical('clip_grad_norm', [0.5, 1.0]),
        
        'epochs': 10 
    }

if __name__ == "__main__":
    run_modpo_tuning_generic(
        finetune_func=run_modpo_finetune,
        config_path="examples/config/ml100k_sasrec_grpo.yaml",
        base_config_path="examples/config/ml100k_sasrec.yaml",
        n_trials=20,
        output_file='modpo_ml100k_optuna_results.csv',
        param_space_func=ml100k_param_space,
        push_best_to_hub=True
    )
