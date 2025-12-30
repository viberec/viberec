from recbole.trainer import Trainer
from recbole.utils import early_stopping, get_model
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any
import copy
import wandb
from viberec.trainers.rl_trainer import RLFinetuneTrainer

class DPOTrainer(RLFinetuneTrainer):
    """
    Implements Teacher-Student Distillation via DPO.
    
    * Student (Policy): Low-Rank Model (d=16) - Trainable
    * Teacher (Reference): High-Rank Model (d=64) - Frozen
    
    Strategy: 
    The Student tries to rank items such that its preference distribution 
    aligns with the Teacher's superior probability manifold.
    """
    def __init__(self, config, model, dataset=None):
        super(DPOTrainer, self).__init__(config, model)
        
        # --- 1. Load the Teacher (Reference Model) ---
        # --- 1. Load the Teacher (Reference Model) ---
        teacher_ckpt_path = config.get('teacher_ckpt_path', None)
        
        # Check if we should download from HF
        teacher_repo_id = config.get('teacher_repo_id', None)
        if teacher_repo_id:
            try:
                from huggingface_hub import hf_hub_download, HfApi
                self.logger.info(f"Downloading Teacher Model from HF Repo: {teacher_repo_id}")
                
                # Logic to find .pth file if specific filename not given (though usually we need structure)
                # Assuming standard structure or just grabbing the first .pth
                # Or even better, let `teacher_ckpt_path` be the filename if provided, else default
                
                api = HfApi()
                files = api.list_repo_files(repo_id=teacher_repo_id)
                pth_files = [f for f in files if f.endswith('.pth')]
                
                if pth_files:
                    # If multiple, maybe pick one? or rely on user? 
                    # For now, pick the first one which is standard in our flow
                    filename = pth_files[0]
                    teacher_ckpt_path = hf_hub_download(repo_id=teacher_repo_id, filename=filename)
                    self.logger.info(f"Downloaded Teacher Checkpoint to: {teacher_ckpt_path}")
                else:
                    self.logger.warning(f"No .pth found in {teacher_repo_id}")

            except Exception as e:
                self.logger.error(f"Failed to download teacher from HF: {e}")
        
        if not teacher_ckpt_path:
             raise ValueError("Teacher Checkpoint not found! Please provide 'teacher_ckpt_path' or 'teacher_repo_id' in config.")

        self.logger.info(f"👨‍🏫 Loading Teacher Model from: {teacher_ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(teacher_ckpt_path, map_location=self.device, weights_only=False)
        teacher_config = checkpoint['config']
        
        # Validate Teacher Architecture
        if teacher_config.get('hidden_size') != 64:
             self.logger.warning(f"⚠️ Teacher model hidden_size is {teacher_config.get('hidden_size')}, expected 64! Ideally it should be larger than student (d={config['hidden_size']}).")
             
        # Initialize Teacher with its own config (d=64) but SAME dataset
        teacher_model_class = get_model(teacher_config['model'])
        self.ref_model = teacher_model_class(teacher_config, dataset).to(self.device)
        # Sanitize state_dict keys (remove _orig_mod. prefix if present from torch.compile)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        self.ref_model.load_state_dict(new_state_dict)
        
        # Freeze Teacher completely
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # --- 2. Config ---
        self.group_size = config.get('group_size', 16) 
        self.beta = config.get('kl_beta', 0.1)      

    def _build_optimizer(self, **kwargs):
        """Force AdamW for stable Transformer Finetuning."""
        params = self.model.parameters()
        learner = self.config.get('learner', 'adam')
        learning_rate = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.01)

        # Always default to AdamW if 'adam' or 'adamw' is requested
        if 'adam' in learner.lower():
            self.logger.info(f"⚡ Using AdamW Optimizer (lr={learning_rate}, decay={weight_decay})")
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = super()._build_optimizer(**kwargs)
            
        return optimizer

    def _predict_topk(self, model, interaction, k):
        """Greedy Top-K prediction."""
        scores = model.full_sort_predict(interaction)
        _, topk_items = torch.topk(scores, k, dim=-1)
        return topk_items.unsqueeze(1) # [Batch, 1, K]

    def get_preference_batch(self, batch_data, group_size=4):
        """
        Generates pairs by pitting Student vs Teacher vs Truth.
        """
        with torch.no_grad():
            # 1. Student Explores (Exploration)
            # Shape: [Batch, G, K]
            student_lists, _ = self._sample_lists(self.model, batch_data, self.k, group_size=group_size)
            
            # 2. Teacher Suggests (Distillation)
            # Shape: [Batch, 1, K]
            ref_list = self._predict_topk(self.ref_model, batch_data, self.k)
            
            # 3. Ground Truth Injection (Robustness)
            # We get the true item ID and expand it to match the list shape K.
            # Shape: [Batch, 1, 1]
            true_item = batch_data[self.config['ITEM_ID_FIELD']].unsqueeze(1).unsqueeze(2)
            
            # Repeat to match sequence length K (Shape: [Batch, 1, K])
            # This creates a list like [TrueItem, TrueItem, TrueItem...] which has NDCG=1.0
            k_dim = student_lists.size(-1)
            true_item_list = true_item.repeat(1, 1, k_dim)

            # 4. Pool Candidates: [Batch, G+2, K]
            # Pool contains: Student Samples + Teacher Sample + Perfect Answer
            candidate_pool = torch.cat([student_lists, ref_list, true_item_list], dim=1)
            
            # 5. Calculate Rewards (NDCG)
            target_items = batch_data[self.config['ITEM_ID_FIELD']]
            all_ndcg = self.calc_ndcg(candidate_pool, target_items)
            
            # Safety checks for -1.0 (invalid items)
            final_scores = torch.where(all_ndcg > 0, all_ndcg, torch.tensor(-1.0, device=self.device))
            
            # 6. Select Winner (Best) vs Loser (Worst)
            # Since 'true_item_list' is in the pool, the max score will almost always be 1.0.
            best_indices = torch.argmax(final_scores, dim=1)
            worst_indices = torch.argmin(final_scores, dim=1)
            
            # Gather IDs
            batch_indices = torch.arange(candidate_pool.size(0), device=self.device)
            chosen_ids = candidate_pool[batch_indices, best_indices]
            rejected_ids = candidate_pool[batch_indices, worst_indices]
            
            # 7. Validity Mask
            chosen_ndcg = all_ndcg[batch_indices, best_indices]
            rejected_ndcg = all_ndcg[batch_indices, worst_indices]
            
            # Valid if Winner is strictly better than Loser
            valid_mask = (chosen_ndcg > rejected_ndcg)
            
            return chosen_ids, rejected_ids, valid_mask

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0.0
        num_valid_batches = 0
        
        iter_data = tqdm(train_data, desc=f"DPO-Robust Epoch {epoch_idx}") if show_progress else train_data

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # 1. Get Preferences (Teacher vs Student vs Truth)
            chosen_ids, rejected_ids, valid_mask = self.get_preference_batch(interaction, self.group_size)
            
            if valid_mask.sum() == 0:
                continue
                
            # Filter Batch to only valid pairs
            valid_interaction = interaction[valid_mask]
            chosen_ids = chosen_ids[valid_mask]
            rejected_ids = rejected_ids[valid_mask]
            
            # 2. Student Log Probs (d=16)
            # "How likely is the Student to pick these items?"
            policy_chosen_logps, _ = self._get_log_probs(self.model, valid_interaction, chosen_ids)
            policy_rejected_logps, _ = self._get_log_probs(self.model, valid_interaction, rejected_ids)
            
            # 3. Teacher Log Probs (d=64) - NO GRAD
            # "How likely would the Teacher be to pick these items?"
            # NOTE: Scalars are dimension-agnostic, so d=16 vs d=64 is fine here.
            with torch.no_grad():
                ref_chosen_logps, _ = self._get_log_probs(self.ref_model, valid_interaction, chosen_ids)
                ref_rejected_logps, _ = self._get_log_probs(self.ref_model, valid_interaction, rejected_ids)

            # 4. DPO Loss Calculation
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            # The DPO Gradient Magic
            logits = self.beta * (policy_logratios - ref_logratios)
            losses = -F.logsigmoid(logits)
            loss = losses.mean()
            
            # 5. Backprop
            loss.backward()
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # Logging
            if self.config.get('log_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': (logits > 0).float().mean().item(),
                    'train/num_valid_pairs': valid_mask.sum().item(),
                    'epoch': epoch_idx
                })

            total_loss += loss.item()
            num_valid_batches += 1

        avg_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0.0
        self.logger.info(f"Epoch {epoch_idx} done. Loss: {avg_loss:.4f} | Valid Batches: {num_valid_batches}")
        return avg_loss
