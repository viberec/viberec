import os
import copy
import logging
import yaml
import torch
from huggingface_hub import hf_hub_download, HfApi
from recbole.config import Config
from recbole.utils import init_seed, init_logger, get_model, get_flops
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.quick_start import load_data_and_model

from viberec.trainer import GRPOTrainer

def run_rl_finetune(finetune_config_path, base_config_path="examples/config/ml100k_sasrec.yaml", push_to_hub=True, **kwargs):
    """
    Runs the RL fine-tuning process using GRPO.
    
    Args:
        finetune_config_path (str): Path to the fine-tuning YAML config.
        base_config_path (str): Path to the base model YAML config.
        **kwargs: Overrides for configuration parameters (for HyperTuning).
    """
    
    # Load Fine-tuning Config First (to get repo_id)
    with open(finetune_config_path, 'r') as f:
         ft_config = yaml.safe_load(f)
         
    repo_id_base = ft_config.get('finetune', {}).get('repo_id', None)
    
    config_file_list = [base_config_path, finetune_config_path]
    
    # --- Standard RecBole Workflow Start ---
    config = Config(
        model=kwargs.get('model', ft_config.get('model', 'SASRec')),
        dataset=kwargs.get('dataset', ft_config.get('dataset', 'ml-100k')),
        config_file_list=config_file_list,
        config_dict={
            'log_wandb': True, 
            'wandb_project': 'viberec',
        }
    )
    
    # Set WandB Name dynamically from Config
    alpha_val = kwargs.get('alpha', ft_config.get('finetune', {}).get('alpha', 0.5))
    run_name = f"{config['dataset']}-{config['model']}-GRPO-Alpha{alpha_val}"
    os.environ["WANDB_NAME"] = run_name
    
    model_name = config['model']
    dataset_name = config['dataset']
    
    # Init seed
    init_seed(config["seed"], config["reproducibility"])
    
    # Init logger
    init_logger(config)
    logger = logging.getLogger()
    
    # Create dataset and dataloaders
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Init model structure
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    
    # Load Pre-trained Weights if provided
    if repo_id_base:
        logger.info(f"Downloading/Loading pre-trained model from HF: {repo_id_base}")
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id_base)
            pth_files = [f for f in files if f.endswith('.pth')]
            
            if pth_files:
                model_path = hf_hub_download(repo_id=repo_id_base, filename=pth_files[0])
                logger.info(f"Loading weights from {model_path}")
                
                # RecBole saves full checkpoint dict
                checkpoint = torch.load(model_path, map_location=config['device'], weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                logger.warning(f"No .pth files found in {repo_id_base}, starting from scratch.")
        except Exception as e:
            logger.error(f"Failed to load model from HF: {e}")
            logger.warning("Starting from scratch...")
            
    # FLOPs
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(f"FLOPs: {flops}")
    # --- Standard RecBole Workflow End ---

    # 2. Inject RL Params into RecBole Config
    # Priority: kwargs > ft_config > defaults
    config['alpha'] = kwargs.get('alpha', ft_config['finetune'].get('alpha', 0.5))
    config['group_size'] = kwargs.get('group_size', ft_config['finetune'].get('group_size', 4))
    config['kl_beta'] = kwargs.get('kl_beta', ft_config['finetune'].get('kl_beta', 0.01))
    config['clip_grad_norm'] = kwargs.get('clip_grad_norm', ft_config['finetune'].get('clip_grad_norm', 1.0))
    config['learning_rate'] = kwargs.get('learning_rate', ft_config['finetune'].get('learning_rate', config['learning_rate']))
    
    config['epochs'] = kwargs.get('epochs', ft_config['finetune'].get('epochs', 3))
    
    print(f"\n=== Starting GRPO Fine-tuning ===")
    
    # 3. Initialize Custom GRPOTrainer
    trainer = GRPOTrainer(config, model, dataset)
    
    # 3.5 Evaluate Baseline (Before Fine-tuning)
    print("\nEvaluating Baseline Model...")
    # Initialize collector for metric calculations (CRITICAL for TailPercentage/Popularity metrics)
    trainer.eval_collector.data_collect(train_data)
    baseline_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config.get('show_progress', True))
    print("Baseline Results:")
    print(baseline_result)
    
    # 4. Train
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data=valid_data,
        show_progress=config.get('show_progress', True),
        saved=True 
    )
    
    print("\n=== Fine-tuning Complete ===")
    
    # 5. Evaluate
    print("Evaluating Fine-tuned Model...")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    print("\nTest Results:")
    print(test_result)
    
    # 6. Upload to Hugging Face
    upload_repo_id = ft_config.get('finetune', {}).get('upload_repo_id', None)
    
    if not upload_repo_id and repo_id_base:
        # Auto-generate upload repo name (Append -GRPO)
        base_name = repo_id_base.split('/')[-1]
        upload_repo_id = f"viberec/{base_name}-GRPO"
        
    if push_to_hub and upload_repo_id:
        print(f"\n=== Uploading to Hugging Face: {upload_repo_id} ===")
        saved_model_path = trainer.saved_model_file
        print(f"Model saved locally at {saved_model_path}")
        
        api = HfApi()
        try:
            api.create_repo(repo_id=upload_repo_id, exist_ok=True, private=False)
            print(f"Repo {upload_repo_id} ready.")
            
            # Upload Model
            api.upload_file(
                path_or_fileobj=saved_model_path,
                path_in_repo=os.path.basename(saved_model_path),
                repo_id=upload_repo_id
            )
            
            # Save and Upload Config
            # Ensure tmp directory exists
            os.makedirs("tmp", exist_ok=True)
            
            config_filename = f"{dataset_name.lower()}_{model_name.lower()}_grpo.yaml"
            output_config_path = os.path.join("tmp", config_filename)
            
            config_dict = config.final_config_dict
            serializable_config = {}
            for k, v in config_dict.items():
                if k == 'data_path': continue
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    serializable_config[k] = v
                else:
                    serializable_config[k] = str(v)
            
            with open(output_config_path, 'w') as f:
                yaml.dump(serializable_config, f, default_flow_style=False)
                
            api.upload_file(
                path_or_fileobj=output_config_path,
                path_in_repo=config_filename,
                repo_id=upload_repo_id
            )
            
            # Generate Model Card
            def format_metrics(metrics_dict):
                return "\n".join([f"- **{k}**: {v}" for k, v in metrics_dict.items()])

            model_card_content = f"""---
tags:
- recommender-system
- {model_name.lower()}
- reinforcement-learning
- grpo
datasets:
- {dataset_name}
---

# {model_name} - {dataset_name} (Finetuned with GRPO)

## Model Description
This model is fine-tuned from [{repo_id_base}](https://huggingface.co/{repo_id_base}) using Group Relative Policy Optimization (GRPO).
The objective was to improve serendipity (Tail Percentage, Low Popularity) while maintaining ranking accuracy (NDCG).

## Training Results
### Baseline (Before Finetune)
{format_metrics(baseline_result)}

### Best Valid Results (GRPO)
{format_metrics(best_valid_result)}

### Test Results (GRPO)
{format_metrics(test_result)}

## RL Hyperparameters
- **Alpha**: {config['alpha']} (Weight for Useful Reward vs Unexpected Reward)
- **KL Beta**: {config['kl_beta']}
- **Group Size**: {config['group_size']}
- **Learning Rate**: {config['learning_rate']}

## Usage
"""
            model_card_path = os.path.join("tmp", "README.md")
            with open(model_card_path, 'w') as f:
                f.write(model_card_content)
                
            api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=upload_repo_id
            )
            
            print(f"Upload complete! Model uploaded to: {upload_repo_id}")
            
        except Exception as e:
            print(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()

    return test_result
