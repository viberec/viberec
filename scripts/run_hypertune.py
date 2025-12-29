import argparse
import sys
import os

# Ensure the parent directory is in sys.path to allow importing viberec modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viberec.hypetune import hypertune

# Import custom trainers here or dynamically load them
from viberec.trainers.dpo_trainer import DPOTrainer
from viberec.trainers.ppo_trainer import PPOTrainer

def main():
    parser = argparse.ArgumentParser(description='Run Viberec Hyperparameter Tuning')
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='Model name')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--config_files', type=str, default=None, help='Config files (space separated)')
    parser.add_argument('--params_file', type=str, default=None, help='Hyperparameter tuning config file (params file)')
    parser.add_argument('--output_file', type=str, default='scripts/output/hyper_tuning.result', help='Output file for tuning results')
    parser.add_argument('--task', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='Task type: pretrain or finetune')
    parser.add_argument('--repo_id', type=str, default=None, help='HuggingFace Repo ID (unused during tuning usually)')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace Token for authentication')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='WandB API Key for logging')
    parser.add_argument('--pretrained_repo_id', type=str, default=None, help='HuggingFace Repo ID to load pre-trained weights from')
    parser.add_argument('--trainer', type=str, default=None, help='Custom Trainer Class Name (e.g., DPOTrainer)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples/trials to run')
    parser.add_argument('--gpus', type=float, default=0, help='GPUs per trial')
    parser.add_argument('--cpus', type=int, default=1, help='CPUs per trial')
    parser.add_argument('--max_t', type=int, default=50, help='Max epochs for ASHA scheduler')
    parser.add_argument('--grace_period', type=int, default=1, help='Grace period for ASHA scheduler')
    parser.add_argument('--reduction_factor', type=int, default=2, help='Reduction factor for ASHA scheduler')
    parser.add_argument('--max_concurrent_trials', type=int, default=None, help='Max concurrent trials (None for unlimited)')
    parser.add_argument('--grid_search', action='store_true', help='Enable Grid Search (force tune.grid_search instead of choice)')

    args, unknown = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # Parse unknown args to config_dict or cli_search_space
    config_dict = {}
    cli_search_space = {}
    import ast
    
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith('--'):
            key = key[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                value = unknown[i+1]
                
                # Try to parse as search space list
                is_search_param = False
                if value.strip().startswith('[') and value.strip().endswith(']'):
                    try:
                        parsed_val = ast.literal_eval(value)
                        if isinstance(parsed_val, list):
                            cli_search_space[key] = parsed_val
                            is_search_param = True
                    except:
                         pass

                if not is_search_param:
                    # Simple type inference
                    if value.lower() == 'true': value = True
                    elif value.lower() == 'false': value = False
                    elif value.lower() == 'none': value = None
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                    config_dict[key] = value
                
                i += 2
            else:
                config_dict[key] = True
                i += 1
        else:
            i += 1

    # Map trainer string to class
    trainer_class = None
    if args.trainer == 'DPOTrainer':
        trainer_class = DPOTrainer
    elif args.trainer == 'PPOTrainer':
        trainer_class = PPOTrainer
        
    hypertune(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        params_file=args.params_file,
        output_file=args.output_file,
        task=args.task,
        trainer_class=trainer_class,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        wandb_api_key=args.wandb_api_key,
        pretrained_repo_id=args.pretrained_repo_id,
        cli_config_dict=config_dict,
        cli_search_space=cli_search_space,
        grid_search=args.grid_search,
        num_samples=args.num_samples,
        gpus=args.gpus,
        cpus=args.cpus,
        max_t=args.max_t,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor,
        max_concurrent_trials=args.max_concurrent_trials
    )

if __name__ == '__main__':
    main()
