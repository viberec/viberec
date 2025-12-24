import os
import logging
import torch
from recbole.config import Config
from viberec.data import load_data
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from huggingface_hub import hf_hub_download
from viberec.huggingface import upload_to_huggingface

def finetune(model_name, dataset_name, config_file_list, trainer_class=None, repo_id=None, hf_token=None, wandb_api_key=None, pretrained_repo_id=None, config_dict=None):
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        
    # 1. Initialize Configuration
    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=config_file_list,
        config_dict=config_dict
    )
    
    # 2. Setup Seed and Logger
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = logging.getLogger()
    
    logger.info(f"Running Fine-tuning for {model_name} on {dataset_name}")
    logger.info(config)

    # 3. Create Dataset and DataLoaders
    dataset, train_data, valid_data, test_data = load_data(config)
    
    # 4. Initialize Model
    model_class = get_model(config['model'])
    model = model_class(config, train_data.dataset).to(config['device'])
    
    # [Optional] Load Pre-trained Weights if specified
    pretrained_path = None
    if pretrained_repo_id:
        logger.info(f"Downloading pre-trained model from Hugging Face Repo: {pretrained_repo_id}")
        try:
             # Look for .pth file usually named like model.pth or similar
             # We assume standard RecBole saving convention, or just try to find the first .pth
             from huggingface_hub import HfApi
             api = HfApi(token=hf_token)
             files = api.list_repo_files(repo_id=pretrained_repo_id)
             pth_files = [f for f in files if f.endswith('.pth')]
             
             if pth_files:
                 filename = pth_files[0]
                 logger.info(f"Found model file: {filename}")
                 pretrained_path = hf_hub_download(repo_id=pretrained_repo_id, filename=filename, token=hf_token)
             else:
                 logger.warning(f"No .pth file found in {pretrained_repo_id}")
        except Exception as e:
            logger.error(f"Failed to download from HF: {e}")

    # Fallback to config path if no HF download or if explicitly set
    if not pretrained_path and 'pretrained_path' in config:
        pretrained_path = config['pretrained_path']
        
    if pretrained_path:
        logger.info(f"Loading pre-trained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config['device'], weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    logger.info(model)
    
    # 5. Initialize Trainer
    if trainer_class:
        logger.info(f"Using custom trainer class: {trainer_class.__name__}")
        trainer = trainer_class(config, model)
    else:
        logger.info("Using default RecBole trainer")
        trainer_class_default = get_trainer(config['MODEL_TYPE'], config['model'])
        trainer = trainer_class_default(config, model)
    
    # 6. Train (Fine-tune)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    # 7. Evaluate
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config['show_progress']
    )
    
    logger.info(f"Best Valid Score: {best_valid_score}")
    logger.info(f"Best Valid Result: {best_valid_result}")
    logger.info(f"Test Result: {test_result}")
    
    # 8. Upload to Hugging Face if repo_id is provided
    if repo_id:
        upload_to_huggingface(
            repo_id=repo_id,
            trainer=trainer,
            config=config,
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=test_result,
            hf_token=hf_token
        )

    return {
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
