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
    Optimized for maximizing NDCG while controlling Serendipity variance.
    """
    return {
        # 1. Learning Rate: Removed 1e-4, Added 8e-5.
        # 1e-4 is risky for ML-1M with G=32 (gradients sum up). 
        # 3e-5 might be too slow for the larger dataset.
        'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 8e-5, 1e-4]),
        
        # 2. Alpha: Removed 0.8, Added 0.95.
        # You explicitly want to IMPROVE NDCG. 
        # 0.8 allows a 20% accuracy drop for serendipity, which hurts your goal.
        # 0.95 forces the model to only pick "Safe" serendipity.
        'alpha': trial.suggest_categorical('alpha', [0.9, 0.95]), 
        
        # 3. Group Size: Kept 16 and 32.
        # ML-1M needs deep sampling (32) to find the "Positive Mutants" in the larger item space.
        # 16 is the safer fallback for speed/VRAM.
        'group_size': trial.suggest_categorical('group_size', [16, 32]),
        
        # 4. KL Beta: Focused on Stability.
        # 0.05 is banned (Too risky for G=32).
        # 0.1 is the sweet spot. 
        # 0.2 is the safety anchor if G=32 causes drift.
        'kl_beta': trial.suggest_categorical('kl_beta', [0.1, 0.15, 0.2]),
        
        # 5. Clip Grad: Fixed to 0.5.
        # In DPO, stable gradients are better than large clipped ones. 
        # Removing 1.0 reduces search space noise.
        'clip_grad_norm': trial.suggest_categorical('clip_grad_norm', [0.5]),
        
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
