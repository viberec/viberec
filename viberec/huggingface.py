import os
import yaml
from huggingface_hub import HfApi

def upload_to_huggingface(repo_id, trainer, config, model_name, dataset_name, metrics, hf_token=None):
    print(f"\nExample: Starting upload to Hugging Face Hub: {repo_id}")
    
    api = HfApi(token=hf_token)
    token = hf_token or HfApi().token
    if not token:
        print("Warning: No Hugging Face token found. Please run `huggingface-cli login` or set HF_TOKEN env var.")
        # Proceeding might fail if repo is private or write access needed, but let's try.

    # Create Repo
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo (might already exist or permission denied): {e}")

    # Prepare temporary directory for artifacts
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 1. Save Config (YAML)
    config_dict = config.final_config_dict
    # Filter non-serializable objects if necessary, though RecBole config is mostly clean.
    serializable_config = {}
    for k, v in config_dict.items():
        if k in ['data_path', 'device']: # Skip local paths or runtime objects
             continue
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
             serializable_config[k] = v
        else:
             serializable_config[k] = str(v)

    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(serializable_config, f, default_flow_style=False)
        
    # 2. Save Model
    # RecBole trainer saves model to `trainer.saved_model_file`. 
    # We can upload that directly or copy it.
    saved_model_path = trainer.saved_model_file
    if not saved_model_path or not os.path.exists(saved_model_path):
        print("Error: Could not find saved model file from trainer.")
        return

    model_filename = os.path.basename(saved_model_path)
    
    # 3. Create Model Card (README.md)
    readme_content = generate_model_card(model_name, dataset_name, metrics, serializable_config)
    readme_path = os.path.join(tmp_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    # Upload all
    try:
        print(f"Uploading config.yaml...")
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.yaml",
            repo_id=repo_id
        )
        
        print(f"Uploading model: {model_filename}...")
        api.upload_file(
            path_or_fileobj=saved_model_path,
            path_in_repo=model_filename,
            repo_id=repo_id
        )
        
        print(f"Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id
        )
        
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"Error during upload: {e}")
    finally:
        # Cleanup tmp dir if needed, or leave for debugging
        pass

def generate_model_card(model_name, dataset_name, metrics, config):
    # Sanitize metrics to ensure they are standard Python types
    sanitized_metrics = {}
    if metrics:
        import numpy as np
        import collections
        for k, v in metrics.items():
            if isinstance(v, (np.integer, int)):
                 sanitized_metrics[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                 sanitized_metrics[k] = float(v)
            else:
                 sanitized_metrics[k] = str(v)

    metrics_str = "\n".join([f"- **{k}**: {v}" for k, v in sanitized_metrics.items()])
    
    yaml_config = yaml.dump(config, default_flow_style=False)
    
    return f"""---
tags:
- viberec
- recommender-system
- {model_name.lower()}
datasets:
- {dataset_name}
metrics:
{yaml.dump(sanitized_metrics, default_flow_style=False)}
---

# {model_name} trained on {dataset_name}


## Model Description
- **Model**: {model_name}
- **Dataset**: {dataset_name}

## Performance
{metrics_str}

## Configuration
```yaml
{yaml_config}
```
"""
