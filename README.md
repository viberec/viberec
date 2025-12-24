# Viberec: Recommender System Framework

This repository uses [RecBole](https://recbole.io/) as a foundation for training and fine-tuning recommender systems, with enhanced support for Hugging Face integration and Hyperparameter tuning via Ray Tune.

The following examples demonstrate how to work with the **ML-100k** dataset using the **SASRec** model.

## Prerequisites

Ensure you have the dependencies installed:
```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

---

## 1. Data Processing

Preprocess the dataset and dataloaders using RecBole and upload them to Hugging Face for faster loading in subsequent steps.

**Command:**
```bash
uv run scripts/process_data.py \
  --dataset ml-100k \
  --config_files scripts/config/ml-100k.yaml \
  --repo_id viberec/ml-100k \
  --hf_token "YOUR_HF_TOKEN"
```

---

## 2. Pretraining

Run the pretraining script to train a base model on the ML-100k dataset. You can optionally upload the model to Hugging Face Hub.

**Command:**
```bash
uv run python scripts/run_pretrain.py \
  --model SASRec \
  --dataset ml-100k \
  --config_files "scripts/config/ml-100k.yaml" \
  --epochs 50 \
  --repo_id "viberec/ml-100k-SASRec" \
  --hf_token "YOUR_HF_TOKEN"
```

**Key Arguments:**
*   `--model`: Model name (e.g., `SASRec`, `BPR`).
*   `--dataset`: Dataset name (e.g., `ml-100k`).
*   `--config_files`: Path to RecBole config files.
*   `--repo_id` (Optional): Hugging Face repository ID to upload the model to.
*   `--hf_token` (Optional): Hugging Face API token (can also be set via `HF_TOKEN` env var).
*   `--wandb_api_key` (Optional): Weights & Biases API key.

---

## 3. Fine-tuning

Fine-tune a pre-trained model (e.g., one downloaded from Hugging Face) using a custom trainer like `DPOTrainer`.

**Command:**
```bash
uv run python scripts/run_finetune.py \
  --model SASRec \
  --dataset ml-100k \
  --config_files "scripts/config/ml-100k.yaml" \
  --pretrained_repo_id "viberec/ml-100k-SASRec" \
  --trainer DPOTrainer \
  --repo_id "viberec/ml-100k-SASRec-DPO" \
  --learning_rate 0.00001 \
  --epochs 10 \
  --hf_token "YOUR_HF_TOKEN"
```

**Key Arguments:**
*   `--pretrained_repo_id`: Hugging Face ID of the base model to load weights from.
*   `--trainer`: Custom trainer class to use (e.g., `DPOTrainer`).
*   `--repo_id`: Hugging Face repository ID to upload the fine-tuned model.
*   **Overrides**: Any CLI argument (like `--learning_rate`, `--epochs`) overrides the config file.

---

## 4. Hyperparameter Tuning

Run hyperparameter optimization using **Ray Tune**. You need to define a parameters file defining the search space.

**Example Params File (`scripts/config/dpo_hyper.test`):**
```text
learning_rate choice [1e-5, 5e-5, 1e-4]
train_batch_size choice [1024, 2048]
```

**Command:**
```bash
uv run python scripts/run_hypertune.py \
  --model SASRec \
  --dataset ml-100k \
  --config_files "scripts/config/ml-100k.yaml" \
  --params_file "scripts/config/dpo_hyper.test" \
  --task finetune \
  --trainer DPOTrainer \
  --pretrained_repo_id "viberec/ml-100k-SASRec" \
  --output_file "scripts/output/hyper_tuning.result" \
  --num_samples 5 \
  --gpus 0 \
  --cpus 1
```

**Key Arguments:**
*   `--task`: Task type, either `pretrain` or `finetune`.
*   `--params_file`: Path to the file defining the hyperparameter search space.
    *   Supported format: `param_name type [values]` (e.g., `lr choice [0.1, 0.01]`).
*   `--num_samples`: Number of trials to run.
*   `--gpus` / `--cpus`: Resources allocated per trial (passed to Ray).
*   `--output_file`: Where to save the tuning results summary.
