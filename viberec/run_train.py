import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduce OneDNN noise

import sys
import yaml
import logging
from huggingface_hub import HfApi, create_repo, hf_hub_download

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, get_flops
from recbole.data.transform import construct_transform
from recbole.quick_start import load_data_and_model

import torch.distributed as dist
import torch

def verify_hf_model(repo_id, filename):
    print(f"\n=== Verifying Model from Hugging Face: {repo_id}/{filename} ===")
    try:
        # Download model
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded model to {downloaded_path}")
        
        # Load and evaluate
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(downloaded_path)
        
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        # Register data statistics for metrics like TailPercentage/AveragePopularity
        trainer.eval_collector.data_collect(train_data)
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        print("Verification successful! Test result from downloaded model:")
        print(test_result)
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_experiment_and_upload(model_name, dataset_name, config_file_list, repo_id=None):
    config = Config(
        model=model_name,
        dataset=dataset_name.lower(),
        config_file_list=config_file_list,
    )
    
    # Initialize distributed process group if needed (fixes dataset download barrier error)
    if dist.is_available() and not dist.is_initialized():
        try:
             # Try initializing with env vars (if run via torchrun)
             dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        except Exception:
             # Fallback to single-process init (if run directly)
             print("Initializing dummy distributed group for single-process mode...")
             dist.init_process_group(
                backend='gloo', 
                init_method='tcp://127.0.0.1:12345', 
                rank=0, 
                world_size=1
             )
    
    # Init seed
    init_seed(config["seed"], config["reproducibility"])
    
    # Init logger
    init_logger(config)
    logger = logging.getLogger()
    
    # Create dataset and dataloaders
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Init model
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    
    # FLOPs
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(f"FLOPs: {flops}")
    
    # Init trainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    
    # Train
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    
    # Evaluate
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )
    
    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")
    
    # === Saving to Hugging Face ===
    
    # Create tmp directory for artifacts
    os.makedirs("tmp", exist_ok=True)
    
    # 1. Save config to yaml
    config_filename = f"{dataset_name.lower().replace('-', '_')}_{model_name.lower()}.yaml"
    output_config_path = os.path.join("tmp", config_filename)
    
    # Prepare serializable config
    config_dict = config.final_config_dict
    serializable_config = {}
    for k, v in config_dict.items():
        if k == 'data_path':
            continue
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)

    with open(output_config_path, 'w') as f:
        yaml.dump(serializable_config, f, default_flow_style=False)
        
    print(f"Config saved to {output_config_path}")
    
    # 2. Upload to HF
    saved_model_path = trainer.saved_model_file
    print(f"Model saved locally at {saved_model_path}")
    
    api = HfApi()
    
    # Determine Repo ID if not provided
    if not repo_id:
        repo_name = f"{dataset_name}-{model_name}"
        # Defaulting to viberec organization as requested
        username = "viberec"
        repo_id = f"{username}/{repo_name}"
        print(f"Targeting Hugging Face Repo ID: {repo_id}")

    if repo_id:
        print(f"preparing to upload to {repo_id}...")
        # Create repo if not exists
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            print(f"Repo {repo_id} ready.")
        except Exception as e:
            print(f"Error creating/accessing repo: {e}")
            
        try:
            print(f"Uploading model to {repo_id}...")
            api.upload_file(
                path_or_fileobj=saved_model_path,
                path_in_repo=os.path.basename(saved_model_path),
                repo_id=repo_id
            )
            
            print(f"Uploading config to {repo_id}...")
            api.upload_file(
                path_or_fileobj=output_config_path,
                path_in_repo=config_filename,
                repo_id=repo_id
            )
            # 3. Create and upload Model Card (README.md)
            print(f"Generating Model Card...")
            
            # Format metrics for the model card
            def format_metrics(metrics_dict):
                return "\n".join([f"- **{k}**: {v}" for k, v in metrics_dict.items()])

            model_card_content = f"""---
tags:
- recommender-system
- {model_name.lower()}
datasets:
- {dataset_name}
---

# {model_name} - {dataset_name}

## Model Description
{model_name} (Self-Attentive Sequential Recommendation) is a sequential recommendation model that uses self-attention mechanisms to model user's historical behavior data.

## Training Results
### Best Valid Results
{format_metrics(best_valid_result)}

### Test Results
{format_metrics(test_result)}

## Configuration
The model was trained with the following configuration:
```yaml
{yaml.dump(serializable_config, default_flow_style=False)}
```

## Usage
"""
            model_card_path = os.path.join("tmp", "README.md")
            with open(model_card_path, 'w') as f:
                f.write(model_card_content)
                
            print(f"Uploading Model Card to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_id
            )
            
            # 4. Verify the uploaded model
            verify_hf_model(repo_id, os.path.basename(saved_model_path))

            print(f"Upload complete! Model uploaded to: {repo_id}")
        except Exception as e:
            print(f"Upload failed: {e}")
    else:
        print("No HF_REPO_ID provided and could not auto-generate, skipping upload.")
