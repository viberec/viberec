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
    def __init__(self, config, model, dataset=None, ref_model=None):
        super(DPOTrainer, self).__init__(config, model)
        
        # --- 1. Load the Teacher (Reference Model) ---
        if ref_model is None:
             raise ValueError("DPOTrainer now requires 'ref_model' to be passed explicitly in __init__.")
             
        self.ref_model = ref_model
        
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
