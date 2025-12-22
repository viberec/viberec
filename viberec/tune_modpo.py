import optuna
import logging
import pandas as pd

def run_modpo_tuning_generic(
    finetune_func,
    config_path, 
    base_config_path, 
    n_trials=10, 
    output_file='modpo_optuna_results.csv',
    param_space_func=None,
    push_best_to_hub=True
):
    """
    Generic Driver for MODPO Hyperparameter Tuning using Optuna.
    
    Args:
        finetune_func (callable): Function that runs the training. Must accept (finetune_config_path, base_config_path, push_to_hub, **params) and return a metrics dict.
        config_path (str): Path to fine-tuning config.
        base_config_path (str): Path to base model config.
        n_trials (int): Number of Optuna trials.
        output_file (str): CSV file to save results.
        param_space_func (callable): Function taking (trial) and returning a dict of parameters to suggest.
        push_best_to_hub (bool): Whether to retrain and upload the best candidate.
    """
    
    def objective(trial):
        """
        Optuna objective function for MODPO.
        Objectives:
        1. Maximize NDCG@10
        2. Minimize AveragePopularity@10
        """
        # Get Params
        if param_space_func:
            params = param_space_func(trial)
        else:
            # Default fallback (Legacy)
            params = {
                'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 5e-5, 1e-4]),
                'alpha': trial.suggest_categorical('alpha', [0.8, 0.9, 0.95]),
                'group_size': trial.suggest_categorical('group_size', [8, 16, 32]),
                'kl_beta': trial.suggest_categorical('kl_beta', [0.1, 0.3]),
                'clip_grad_norm': trial.suggest_categorical('clip_grad_norm', [0.5, 1.0]),
                'epochs': 10 
            }
        
        print(f"\n[Optuna Trial {trial.number}] Params: {params}")
        
        try:
            # Run MODPO Finetuning
            test_result = finetune_func(
                finetune_config_path=config_path, 
                base_config_path=base_config_path,
                push_to_hub=False,
                **params
            )
            
            ndcg = test_result.get('ndcg@10', 0.0)
            avg_pop = test_result.get('averagepopularity@10', 9999.0)
            
            print(f"[Optuna Trial {trial.number}] Result: NDCG={ndcg}, AvgPop={avg_pop}")
            
            return ndcg, avg_pop
            
        except Exception as e:
            print(f"[Optuna Trial {trial.number}] Failed: {e}")
            import traceback
            traceback.print_exc()
            # return worst case
            return 0.0, 9999.0

    # Directions: Maximize NDCG, Minimize AvgPop
    study = optuna.create_study(directions=["maximize", "minimize"])
    
    print(f"Starting MODPO Optuna Tuning for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n=== Optuna Optimization Complete ===")
    print(f"Number of finished trials: {len(study.trials)}")
    
    # Pareto Front
    print("\nPareto Front (Best Trade-offs):")
    best_trials = study.best_trials
    
    for t in best_trials:
        print(f"  Trial #{t.number}: NDCG={t.values[0]:.4f}, Pop={t.values[1]:.2f}, Params={t.params}")
        
    # Export
    df = study.trials_dataframe()
    df.to_csv(output_file)
    print(f"\nAll results saved to {output_file}")
    
    # Select Best Candidate (Highest NDCG on Pareto Front)
    if not best_trials:
        return study
        
    best_candidate = max(best_trials, key=lambda t: t.values[0])
    
    print(f"\n=== Best Candidate Selected for Upload (Trial #{best_candidate.number}) ===")
    print(f"Params: {best_candidate.params}")
    
    if push_best_to_hub:
        print("\nRetraining and Uploading Best Candidate...")
        try:
            finetune_func(
                finetune_config_path=config_path,
                base_config_path=base_config_path,
                push_to_hub=True,
                **best_candidate.params
            )
            print("Final Best Model Uploaded Successfully.")
        except Exception as e:
            print(f"Final upload failed: {e}")
    
    return study
