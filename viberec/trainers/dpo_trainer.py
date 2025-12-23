from recbole.trainer import Trainer
from recbole.utils import early_stopping
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
    Implements Online DPO for RecBole.
    Selection Strategy: Pure NDCG improvement.
    Loss Strategy: Standard DPO (Binary Preference) with Reference Model Constraints.
    """
    def __init__(self, config, model, dataset=None):
        super(DPOTrainer, self).__init__(config, model)
        
        # 1. Reference Model (Deep Copy & Freeze)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # 2. Config
        self.group_size = config.get('group_size', 8) 
        self.beta = config.get('kl_beta', 0.1)        
        
        # Inherited from RLFinetuneTrainer: k, clip_grad_norm, discounts

    def _predict_topk(self, model, interaction, k):
        """Greedy Top-K for Baseline."""
        scores = model.full_sort_predict(interaction)
        _, topk_items = torch.topk(scores, k, dim=-1)
        return topk_items.unsqueeze(1) # [Batch, 1, K]

    def get_preference_batch(self, batch_data, group_size=4):
        with torch.no_grad():
            # 1. Sample Lists (Student + Reference)
            # Use inherited _sample_lists (args: model, interaction, k, group_size)
            student_lists, _ = self._sample_lists(self.model, batch_data, self.k, group_size=group_size)
            ref_list = self._predict_topk(self.ref_model, batch_data, self.k)
            
            # Pool: [Batch, G+1, K]
            candidate_pool = torch.cat([student_lists, ref_list], dim=1)
            
            # 2. Raw Metrics
            target_items = batch_data[self.config['ITEM_ID_FIELD']]
            all_ndcg = self.calc_ndcg(candidate_pool, target_items)
            
            # 3. Pure NDCG Score
            final_scores = all_ndcg
            
            # Safety checks
            final_scores = torch.where(all_ndcg > 0, final_scores, torch.tensor(-1.0, device=self.device))
            
            # 4. Select Winner/Loser
            best_indices = torch.argmax(final_scores, dim=1)
            worst_indices = torch.argmin(final_scores, dim=1)
            
            # Gather IDs
            batch_indices = torch.arange(candidate_pool.size(0), device=self.device)
            chosen_ids = candidate_pool[batch_indices, best_indices]
            rejected_ids = candidate_pool[batch_indices, worst_indices]
            
            # --- 5. The Mask ---
            # Get values for Winner
            chosen_ndcg = all_ndcg[batch_indices, best_indices]
            
            # Get values for Loser
            rejected_ndcg = all_ndcg[batch_indices, worst_indices]
            
            # Condition: NDCG Improvement
            is_ndcg_improved = (chosen_ndcg > rejected_ndcg)
            valid_mask = is_ndcg_improved
            
            return chosen_ids, rejected_ids, valid_mask

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0.0
        num_valid_batches = 0
        
        iter_data = tqdm(train_data, desc=f"DPO Epoch {epoch_idx}") if show_progress else train_data

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # 1. Get Preferences
            chosen_ids, rejected_ids, valid_mask = self.get_preference_batch(interaction, self.group_size)
            
            if valid_mask.sum() == 0:
                continue
                
            # Filter Batch
            valid_interaction = interaction[valid_mask]
            chosen_ids = chosen_ids[valid_mask]
            rejected_ids = rejected_ids[valid_mask]
            
            # 2. Policy Log Probs (Student)
            # _get_log_probs returns (sum_log_probs, all_log_probs_dist)
            policy_chosen_logps, _ = self._get_log_probs(self.model, valid_interaction, chosen_ids)
            policy_rejected_logps, _ = self._get_log_probs(self.model, valid_interaction, rejected_ids)
            
            # 3. Ref Log Probs (Teacher) - NO GRAD
            with torch.no_grad():
                ref_chosen_logps, _ = self._get_log_probs(self.ref_model, valid_interaction, chosen_ids)
                ref_rejected_logps, _ = self._get_log_probs(self.ref_model, valid_interaction, rejected_ids)

            # 4. DPO Loss Calculation
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            # The 'logits' for the sigmoid
            logits = self.beta * (policy_logratios - ref_logratios)
            
            # We want to maximize log_sigmoid(logits) -> minimize -log_sigmoid(logits)
            losses = -F.logsigmoid(logits)
            loss = losses.mean()
            
            loss.backward()
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # Logging
            if self.config.get('log_wandb', False):
                
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ccuracy': (logits > 0).float().mean().item(),
                    'train/num_valid_pairs': valid_mask.sum().item(),
                    'epoch': epoch_idx
                })

            total_loss += loss.item()
            num_valid_batches += 1

        if num_valid_batches > 0:
            avg_loss = total_loss / num_valid_batches
        else:
            avg_loss = 0.0
            
        self.logger.info(f"Epoch {epoch_idx} done. Loss: {avg_loss:.4f} | Valid Batches: {num_valid_batches}")
        return avg_loss
