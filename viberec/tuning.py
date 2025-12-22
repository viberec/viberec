import optuna
import logging
from viberec.run_finetune import run_rl_finetune

def objective(trial, config_path, base_config_path):
    """
    Optuna objective function for Multi-Objective Optimization.
    Objectives:
    1. Maximize NDCG@10
    2. Minimize AveragePopularity@10
    """
    params = {
        # Lower learning rates for stability (1e-5 found best in ML100k MODPO tuning)
        'learning_rate': trial.suggest_categorical('learning_rate', [5e-6, 1e-5]),
        # Alpha 0.5 performed best for MODPO in ML100k tuning (Best Trade-off)
        'alpha': trial.suggest_categorical('alpha', [0.7, 0.8, 0.9, 0.95]),
        # Group size 16 was superior to 8.
        'group_size': trial.suggest_categorical('group_size', [16, 32]),
        # KL Beta 0.3 was optimal, 0.1 also effective.
        'kl_beta': trial.suggest_categorical('kl_beta', [0.1, 0.3]),
        'epochs': 5 
    }
    
    print(f"\n[Optuna Trial {trial.number}] Params: {params}")
    
    try:
        # Run Finetuning
        # run_rl_finetune returns a dictionary of test results
        # We suppress output to keep logs clean(er) if possible, but run_rl_finetune prints a lot.
        test_result = run_rl_finetune(
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
        # Return worst case: Low NDCG, High Pop
        return 0.0, 9999.0

def run_bayes_tuning(config_path="examples/config/ml100k_sasrec_grpo.yaml", base_config_path="examples/config/ml100k_sasrec.yaml", n_trials=20, output_file='optuna_results.csv'):
    # Directions: Maximize NDCG, Minimize AvgPop
    study = optuna.create_study(directions=["maximize", "minimize"])
    
    print(f"Starting Optuna Multi-Objective Optimization for {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, config_path, base_config_path), n_trials=n_trials)
    
    print("\n=== Optuna Optimization Complete ===")
    print(f"Number of finished trials: {len(study.trials)}")
    
    # Pareto Front
    print("\nPareto Front (Best Trade-offs):")
    best_trials = study.best_trials
    
    results = []
    for t in best_trials:
        print(f"  Trial #{t.number}: NDCG={t.values[0]:.4f}, Pop={t.values[1]:.2f}, Params={t.params}")
        results.append({
            'trial_id': t.number,
            'ndcg': t.values[0],
            'pop': t.values[1],
            'params': t.params
        })
        
    # Export all results to CSV
    df = study.trials_dataframe()
    df.to_csv(output_file)
    print(f"\nAll results saved to {output_file}")
    
    # Select Best Candidate (Highest NDCG on Pareto Front) for final upload
    best_candidate = max(best_trials, key=lambda t: t.values[0])
    print(f"\n=== Best Candidate Selected for Upload (Trial #{best_candidate.number}) ===")
    print(f"Params: {best_candidate.params}")
    print(f"Metrics: NDCG={best_candidate.values[0]}, AvgPop={best_candidate.values[1]}")
    
    print("\nRetraining and Uploading Best Candidate...")
    try:
        run_rl_finetune(
            finetune_config_path=config_path,
            base_config_path=base_config_path,
            push_to_hub=True,
            **best_candidate.params
        )
        print("Final Best Model Uploaded Successfully.")
    except Exception as e:
        print(f"Final upload failed: {e}")
    
    return study
