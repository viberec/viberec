import argparse
import os
import sys
import yaml
import logging
import torch
import torch.distributed as dist
from huggingface_hub import HfApi, create_repo

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, get_flops
from recbole.data.transform import construct_transform
from viberec.huggingface import upload_to_huggingface


def pretrain(model_name, dataset_name, config_file_list, repo_id=None, hf_token=None, wandb_api_key=None, config_dict=None):
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key

    # Patch torch.distributed.barrier to avoid errors in single-node/non-distributed runs
    # when RecBole tries to use it during dataset download.
    if not dist.is_initialized():
        dist.barrier = lambda *args, **kwargs: None
    
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
    
    logger.info(f"Running {model_name} on {dataset_name}")
    logger.info(config)

    # 3. Create Dataset and DataLoaders
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 4. Initialize Model
    model_class = get_model(config['model'])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # 5. Initialize Trainer
    trainer_class = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_class(config, model)
    
    # 6. Train
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
