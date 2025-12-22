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
import torch.distributed as dist

from viberec.modapo_trainer import MODPOTrainer

def run_modpo_finetune(finetune_config_path, base_config_path="examples/config/ml100k_sasrec.yaml", push_to_hub=True, **kwargs):
    """
    Runs the RL fine-tuning process using MODPO.
    """
    
    # Load Fine-tuning Config First (to get repo_id)
    with open(finetune_config_path, 'r') as f:
         ft_config = yaml.safe_load(f)
         
    # Support both flat config (new) and nested 'finetune' config (old)
    finetune_section = ft_config.get('finetune', {})
    if not isinstance(finetune_section, dict):
        finetune_section = {}
        
    def get_param(key, default):
        # Priority: kwargs > ft_config (flat) > ft_config['finetune'] > default
        val = kwargs.get(key)
        if val is not None: return val
        val = ft_config.get(key)
        if val is not None: return val
        val = finetune_section.get(key)
        if val is not None: return val
        return default

    repo_id_base = get_param('repo_id', None)
    
    # Initialize distributed process group if needed
    if dist.is_available() and not dist.is_initialized():
        try:
             dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        except Exception:
             print("Initializing dummy distributed group for single-process mode...")
             dist.init_process_group(
                 backend='gloo', 
                 init_method='tcp://127.0.0.1:12345', 
                 rank=0, 
                 world_size=1
             )

    config_file_list = [base_config_path, finetune_config_path]
    
    # --- Standard RecBole Workflow Start ---
    base_model_cfg = {}
    if base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            base_model_cfg = yaml.safe_load(f) or {}

    dataset_arg = kwargs.get('dataset', ft_config.get('dataset', base_model_cfg.get('dataset', None)))
    model_arg = kwargs.get('model', ft_config.get('model', base_model_cfg.get('model', 'SASRec')))

    config = Config(
        model=model_arg,
        dataset=dataset_arg, 
        config_file_list=config_file_list,
        config_dict={
            'log_wandb': True, 
            'wandb_project': 'viberec',
        }
    )
    
    # Set WandB Name dynamically
    alpha_val = get_param('alpha', 0.5)
    run_name = f"{config['dataset']}-{config['model']}-MODPO-Alpha{alpha_val}"
    os.environ["WANDB_NAME"] = run_name
    
    model_name = config['model']
    dataset_name = config['dataset']
    
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    
    # Load Pre-trained Weights
    if repo_id_base:
        logger.info(f"Downloading/Loading pre-trained model from HF: {repo_id_base}")
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id_base)
            pth_files = [f for f in files if f.endswith('.pth')]
            
            if pth_files:
                model_path = hf_hub_download(repo_id=repo_id_base, filename=pth_files[0])
                logger.info(f"Loading weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=config['device'], weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                logger.warning(f"No .pth files found in {repo_id_base}, starting from scratch.")
        except Exception as e:
            logger.error(f"Failed to load model from HF: {e}")
            logger.warning("Starting from scratch...")
            
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(f"FLOPs: {flops}")
    # --- Standard RecBole Workflow End ---

    # 2. Inject RL Params
    def resolve(key, default):
        # Priority: kwargs > ft_config (flat) > ft_config['finetune'] > config > default
        if key in kwargs: return kwargs[key]
        
        val = ft_config.get(key)
        if val is not None: return val
        
        val = finetune_section.get(key)
        if val is not None: return val
        
        if key in config: return config[key]
        
        return default

    config['alpha'] = resolve('alpha', 0.5)
    config['group_size'] = resolve('group_size', 4)
    config['kl_beta'] = resolve('kl_beta', 0.01)
    config['clip_grad_norm'] = resolve('clip_grad_norm', 1.0)
    config['learning_rate'] = resolve('learning_rate', config['learning_rate'])
    config['epochs'] = resolve('epochs', 3)
    
    print(f"\n=== Starting MODPO Fine-tuning ===")
    
    # 3. Initialize MODPOTrainer
    trainer = MODPOTrainer(config, model, dataset)
    
    # 3.5 Evaluate Baseline
    print("\nEvaluating Baseline Model...")
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
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    print("\nTest Results:")
    print(test_result)
    
    # 6. Upload to Hugging Face
    upload_repo_id = ft_config.get('finetune', {}).get('upload_repo_id', None)
    
    if not upload_repo_id and repo_id_base:
        base_name = repo_id_base.split('/')[-1]
        upload_repo_id = f"viberec/{base_name}-MODPO"
        
    if push_to_hub and upload_repo_id:
        print(f"\n=== Uploading to Hugging Face: {upload_repo_id} ===")
        saved_model_path = trainer.saved_model_file
        print(f"Model saved locally at {saved_model_path}")
        
        api = HfApi()
        try:
            api.create_repo(repo_id=upload_repo_id, exist_ok=True, private=False)
            print(f"Repo {upload_repo_id} ready.")
            
            api.upload_file(
                path_or_fileobj=saved_model_path,
                path_in_repo=os.path.basename(saved_model_path),
                repo_id=upload_repo_id
            )
            
            os.makedirs("tmp", exist_ok=True)
            config_filename = f"{dataset_name.lower()}_{model_name.lower()}_modpo.yaml"
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
- modpo
datasets:
- {dataset_name}
---

# {model_name} - {dataset_name} (Finetuned with MODPO)

## Model Description
This model is fine-tuned from [{repo_id_base}](https://huggingface.co/{repo_id_base}) using Multi-Objective Direct Preference Optimization (MODPO).

## Training Results
### Baseline (Before Finetune)
{format_metrics(baseline_result)}

### Best Valid Results (MODPO)
{format_metrics(best_valid_result)}

### Test Results (MODPO)
{format_metrics(test_result)}

## RL Hyperparameters
- **Alpha**: {config['alpha']}
- **KL Beta**: {config['kl_beta']}
- **Group Size**: {config['group_size']}
- **Learning Rate**: {config['learning_rate']}
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
