import logging
import os
import yaml
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from recbole.config import Config
from recbole.utils import init_logger, init_seed
from viberec.pretrain import pretrain
from viberec.finetune import finetune

def parse_params_file_to_ray_space(params_file):
    """
    Parses a simple RecBole-style params file to Ray Tune search space.
    Supports:
    - parameter choice [a, b, c]
    - parameter uniform [low, high] (mapped to tune.uniform)
    - parameter quniform [low, high, q]
    - parameter loguniform [low, high]
    """
    space = {}
    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            # Simple parsing logic
            # Format: name type [values]
            parts = line.split(' ', 2)
            if len(parts) < 3:
                continue
                
            name = parts[0]
            action = parts[1]
            values_str = parts[2]
            
            # Remove brackets
            values_str = values_str.strip('[]')
            
            if action == 'choice':
                # Split by comma, strip quotes
                vals = [v.strip().strip("'").strip('"') for v in values_str.split(',')]
                # Try to convert to numbers
                parsed_vals = []
                for v in vals:
                    try:
                        if '.' in v or 'e' in v:
                            parsed_vals.append(float(v))
                        else:
                            parsed_vals.append(int(v))
                    except:
                        parsed_vals.append(v)
                space[name] = tune.choice(parsed_vals)
                
            elif action == 'uniform':
                vals = [float(v.strip()) for v in values_str.split(',')]
                space[name] = tune.uniform(vals[0], vals[1])
                
            elif action == 'loguniform':
                vals = [float(v.strip()) for v in values_str.split(',')]
                space[name] = tune.loguniform(vals[0], vals[1])
                
            elif action == 'quniform':
                vals = [float(v.strip()) for v in values_str.split(',')]
                space[name] = tune.quniform(vals[0], vals[1], vals[2])
                
    return space

def hypertune(model, dataset, config_file_list, params_file, output_file='hyper_tuning.result',
              task='pretrain', 
              trainer_class=None, 
              repo_id=None, 
              hf_token=None, 
              wandb_api_key=None,
              pretrained_repo_id=None,
              cli_config_dict=None,
              num_samples=1,
              gpus=0,
              cpus=1,
              max_t=50,
              grace_period=1,
              reduction_factor=2):
              
    # Convert config files to absolute paths to avoid FileNotFoundError in Ray workers
    if config_file_list:
        config_file_list = [os.path.abspath(f) for f in config_file_list]
              
    # 1. Parse Search Space
    config_space = parse_params_file_to_ray_space(params_file)
    
    # 2. Define Trainable
    def train_func(ray_config):
        # Merge CLI config with Ray config
        final_config_dict = {}
        if cli_config_dict:
             final_config_dict.update(cli_config_dict)
        final_config_dict.update(ray_config)
        
        # RecBole expects string args for Config usually, but dict works too.
        # However, we need to handle logging carefully in Ray.
        
        # We need to re-init logger per trial or suppress it? 
        # RecBole logger writes to file. Ray captures stdout/stderr.
        
        try:
            if task == 'finetune':
                 result = finetune(
                    model_name=model,
                    dataset_name=dataset,
                    config_file_list=config_file_list,
                    trainer_class=trainer_class,
                    repo_id=None, 
                    hf_token=hf_token,          
                    wandb_api_key=None, 
                    pretrained_repo_id=pretrained_repo_id,
                    config_dict=final_config_dict
                 )
            else:
                 result = pretrain(
                    model_name=model,
                    dataset_name=dataset,
                    config_file_list=config_file_list,
                    repo_id=None,
                    hf_token=hf_token,
                    wandb_api_key=None,
                    config_dict=final_config_dict
                 )
            
            # Report to Ray
            # RecBole returns {'best_valid_score': ..., ...}
            metrics = {'valid_score': result['best_valid_score']}
            if result.get('best_valid_result'):
                metrics.update(result['best_valid_result'])
            yield metrics
            
        except Exception as e:
            print(f"Trial failed: {e}")
            raise e

    # 3. Run Tune
    logger = logging.getLogger()
    logger.info(f"Starting Ray Tune for {model} on {dataset}")
    
    # Use ASHAScheduler for efficient early stopping
    scheduler = ASHAScheduler(
        metric="valid_score",
        mode="max",
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor
    )
    
    analysis = tune.run(
        train_func,
        config=config_space,
        num_samples=num_samples, 
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        scheduler=scheduler
    )
    
    best_trial = analysis.get_best_trial(metric="valid_score", mode="max")
    best_config = best_trial.config
    print("Best Config: ", best_config)
    print("Best Valid Score: ", best_trial.last_result['valid_score'])
    
    # Save results
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as f:
        f.write(f"Best Config: {best_config}\n")
        best_trial = analysis.get_best_trial(metric="valid_score", mode="max")
        if best_trial:
            f.write(f"Best Result: {best_trial.last_result}\n")
