import os
from viberec.modpo_finetune import run_modpo_finetune

if __name__ == "__main__":
    run_modpo_finetune(
        finetune_config_path="examples/config/ml100k_sasrec_grpo.yaml",
        base_config_path="examples/config/ml100k_sasrec.yaml"
    )


