import argparse
import sys
import os

# Ensure the parent directory is in sys.path to allow importing viberec modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viberec.finetune import finetune
# Import custom trainers here or dynamically load them
from viberec.trainers.dpo_trainer import DPOTrainer

def main():
    parser = argparse.ArgumentParser(description='Run Viberec Fine-tuning')
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='Model name')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--config_files', type=str, default=None, help='Config files (space separated)')
    parser.add_argument('--repo_id', type=str, default=None, help='HuggingFace Repo ID to push to')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace Token for authentication')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='WandB API Key for logging')
    parser.add_argument('--pretrained_repo_id', type=str, default=None, help='HuggingFace Repo ID to load pre-trained weights from')
    parser.add_argument('--trainer', type=str, default=None, help='Custom Trainer Class Name (e.g., DPOTrainer)')

    args, unknown = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # Parse unknown args to config_dict
    config_dict = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith('--'):
            key = key[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                value = unknown[i+1]
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
    # Add other trainers here if needed
    elif args.trainer:
        print(f"Warning: Trainer {args.trainer} not explicitly mapped in script. Trying to load default.")

    finetune(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file_list=config_file_list,
        trainer_class=trainer_class,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        wandb_api_key=args.wandb_api_key,
        pretrained_repo_id=args.pretrained_repo_id,
        config_dict=config_dict
    )

if __name__ == '__main__':
    main()
