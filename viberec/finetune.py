import os
import logging
import torch
from recbole.config import Config
from viberec.data import load_data
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from huggingface_hub import hf_hub_download, HfApi
from viberec.huggingface import upload_to_huggingface

def finetune(model_name, dataset_name, config_file_list, trainer_class=None, repo_id=None, hf_token=None, wandb_api_key=None, pretrained_repo_id=None, config_dict=None):
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key

    # [New] If config_file_list is empty, try to load config.yaml from the pretrained repo
    if not config_file_list and pretrained_repo_id:
        logging.info(f"No local config provided. Attempting to download config.yaml from {pretrained_repo_id}")
        try:
            # from huggingface_hub import HfApi, hf_hub_download # Removed to avoid shadowing
            config_path = hf_hub_download(repo_id=pretrained_repo_id, filename="config.yaml", token=hf_token)
            logging.info(f"Downloaded config to: {config_path}")
            
            if config_file_list is None:
                config_file_list = []
            config_file_list.append(config_path)
            
        except Exception as e:
            logging.warning(f"Could not download config.yaml from {pretrained_repo_id}: {e}")
        
    if config_dict is None:
        config_dict = {}

        
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
             # from huggingface_hub import HfApi # Removed
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
    
    # Save original metrics
    # 5. Initialize Trainer
    
    # Save original metrics
    original_metrics = config['metrics']
    
    # --- Baseline Evaluation (Safe Metrics) ---
    config['metrics'] = ['NDCG', 'Hit', 'AveragePopularity']
    
    import inspect
    logger.info("Using default RecBole trainer for baseline check" if not trainer_class else f"Using {trainer_class.__name__} for baseline check")
    baseline_trainer_cls = trainer_class if trainer_class else get_trainer(config['MODEL_TYPE'], config['model'])
    
    # Check if trainer accepts dataset
    sig = inspect.signature(baseline_trainer_cls.__init__)
    if 'dataset' in sig.parameters:
        baseline_trainer = baseline_trainer_cls(config, model, dataset=dataset)
    else:
        baseline_trainer = baseline_trainer_cls(config, model)
        
    # Manual Collection for AveragePopularity
    from recbole.evaluator import Collector
    baseline_trainer.eval_collector = Collector(config)
    baseline_trainer.eval_collector.data_collect(train_data)
    
    logger.info("Evaluating model before fine-tuning (Baseline)...")
    baseline_result = baseline_trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(f"Baseline Result: {baseline_result}")
    
    # --- Restore Full Metrics ---
    config['metrics'] = original_metrics
    
    # Re-initialize Trainer with full config for Fine-tuning
    if trainer_class:
        logger.info(f"Using custom trainer class: {trainer_class.__name__}")
        cls = trainer_class
    else:
        logger.info("Using default RecBole trainer")
        cls = get_trainer(config['MODEL_TYPE'], config['model'])
        
    sig = inspect.signature(cls.__init__)
    if 'dataset' in sig.parameters:
        trainer = cls(config, model, dataset=dataset)
    else:
        trainer = cls(config, model)
    
    # [New] Evaluate Teacher Model (if available) - Before Training
    teacher_result = {}
    if hasattr(trainer, 'ref_model'):
        logger.info("Evaluating Teacher (Reference Model) before training...")
        
        teacher_trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
        teacher_trainer = teacher_trainer_cls(config, trainer.ref_model)
        
        # Manual Collection for AveragePopularity
        from recbole.evaluator import Collector
        teacher_trainer.eval_collector = Collector(config)
        teacher_trainer.eval_collector.data_collect(train_data)
        
        teacher_result = teacher_trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
        logger.info(f"Teacher Result: {teacher_result}")

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
    
    # --- Compare with Teacher (if available) ---
    if teacher_result:
        teacher_improvement = {}
        for metric, score in test_result.items():
            if metric in teacher_result and teacher_result[metric] != 0:
                if 'popularity' in metric.lower():
                     imp = ((teacher_result[metric] - score) / teacher_result[metric]) * 100
                else:
                     imp = ((score - teacher_result[metric]) / teacher_result[metric]) * 100
                teacher_improvement[metric] = f"{imp:.2f}%"
            else:
                 teacher_improvement[metric] = "N/A"
        logger.info(f"Improvement over Teacher: {teacher_improvement}")

    # Calculate Improvement
    improvement = {}
    for metric, score in test_result.items():
        # Only compare if metric exists in baseline (i.e., NDCG, Hit)
        if metric in baseline_result and baseline_result[metric] != 0:
            if 'popularity' in metric.lower():
                 imp = ((baseline_result[metric] - score) / baseline_result[metric]) * 100
            else:
                 imp = ((score - baseline_result[metric]) / baseline_result[metric]) * 100
            improvement[metric] = f"{imp:.2f}%"
        else:
             improvement[metric] = "N/A"
             
    logger.info(f"Improvement over Baseline: {improvement}")
    
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
