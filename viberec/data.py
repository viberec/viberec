import os
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger
from viberec.huggingface import upload_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
import torch.distributed as dist


def load_data(config):
    """
    Load dataset and prepare dataloaders based on the configuration.

    Args:
        config (Config): The RecBole config object.

    Returns:
        tuple: (dataset, train_data, valid_data, test_data)
    """
    dataset_name = config['dataset']
    logger = logging.getLogger()
    
    # Patch torch.distributed.barrier to avoid errors in single-node/non-distributed runs
    if not dist.is_initialized():
        dist.barrier = lambda *args, **kwargs: None

    # Update config to ensure we save if we create from scratch
    config['save_dataset'] = True
    config['save_dataloaders'] = False
    
    # Attempt to download processed data from Hugging Face
    checkpoint_dir = config['checkpoint_dir']
    dataset_file = f"{dataset_name}.pth"
    
    repo_id = f"viberec/{dataset_name}"
    
    try:
        logger.info(f"Checking for existing processed data in Hugging Face: {repo_id}")
        hf_hub_download(repo_id=repo_id, filename=dataset_file, local_dir=checkpoint_dir, repo_type='dataset')
        logger.info(f"Downloaded processed data to {checkpoint_dir}")
        
        # Update config to look for these specific files
        config['dataset_save_path'] = os.path.join(checkpoint_dir, dataset_file)
        
    except Exception as e:
        logger.info(f"Could not download processed data from {repo_id} (might not exist). Proceeding to create locally. Error: {e}")

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data


def preprocess_and_upload(
    dataset_name,
    repo_id,
    model_name='SASRec',
    config_file_list=None,
    config_dict=None,
    hf_token=None
):
    """
    Preprocess data and dataloader using RecBole and upload to Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'ml-100k').
        repo_id (str): Hugging Face repository ID (e.g., 'username/dataset-name').
        model_name (str): Model name to use for dataloader construction (default: 'SASRec').
        config_file_list (list): List of config files.
        config_dict (dict): Dictionary of config parameters.
        hf_token (str): Hugging Face token.
    """
    # Patch torch.distributed.barrier to avoid errors in single-node/non-distributed runs
    if not dist.is_initialized():
        dist.barrier = lambda *args, **kwargs: None

    # 1. Initialize Config
    if config_dict is None:
        config_dict = {}
    
    # Force saving
    config_dict.update({
        'dataset': dataset_name,
        'model': model_name,
        'save_dataset': True,
        'save_dataloaders': True,
    })
    
    config = Config(model=model_name, dataset=dataset_name, config_file_list=config_file_list, config_dict=config_dict)
    init_logger(config)
    logger = logging.getLogger()
    
    logger.info(f"Preprocessing dataset: {dataset_name} for model: {model_name}")
    
    # 2. Create Dataset (and save)
    dataset = create_dataset(config)
    
    # 3. Create Dataloaders (and save)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 4. Locate and Rename Saved Files
    saved_files = []
    checkpoint_dir = config["checkpoint_dir"]
    
    # --- Process Dataset File ---
    # RecBole default save name
    dataset_class_name = dataset.__class__.__name__
    default_dataset_file = os.path.join(checkpoint_dir, f'{dataset_name}-{dataset_class_name}.pth')
    
    # Desired name
    target_dataset_file = os.path.join(checkpoint_dir, f'{dataset_name}.pth')
    
    if os.path.exists(default_dataset_file):
        logger.info(f"Renaming {default_dataset_file} to {target_dataset_file}")
        os.rename(default_dataset_file, target_dataset_file)
        saved_files.append(target_dataset_file)
    elif os.path.exists(target_dataset_file):
        # Already renamed or saved there (if config worked partly)
        saved_files.append(target_dataset_file)
    else:
        logger.warning(f"Expected dataset file not found at {default_dataset_file}")

    # --- Process Dataloader File ---
    # RecBole default save name
    default_dataloader_file = os.path.join(checkpoint_dir, f'{dataset_name}-for-{model_name}-dataloader.pth')
    
    # Desired name
    target_dataloader_file = os.path.join(checkpoint_dir, f'{dataset_name}-dataloader.pth')
    
    if os.path.exists(default_dataloader_file):
        logger.info(f"Renaming {default_dataloader_file} to {target_dataloader_file}")
        os.rename(default_dataloader_file, target_dataloader_file)
        saved_files.append(target_dataloader_file)
    elif os.path.exists(target_dataloader_file):
        saved_files.append(target_dataloader_file)
    else:
        logger.warning(f"Expected dataloader file not found at {default_dataloader_file}")
        
    if not saved_files:
        logger.error("No files found to upload.")
        return

    # 5. Upload to Hugging Face
    logger.info(f"Uploading files to {repo_id}: {saved_files}")
    upload_dataset(repo_id, saved_files, dataset_name, config=config, hf_token=hf_token, dataset_stats=dataset)
