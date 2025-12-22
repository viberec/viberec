from recbole.trainer import Trainer
from recbole.utils import early_stopping
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any

from viberec.rewards import DeltaRewardCalculator

class MODPOTrainer(Trainer):
    """
    Implements Online DPO for RecBole.
    Selection Strategy: Alpha-weighted Rank (NDCG vs Serendipity).
    Loss Strategy: Standard DPO (Binary Preference) with Reference Model Constraints.
    """
    def __init__(self, config, model, dataset=None):
        super(MODPOTrainer, self).__init__(config, model)
        
        # 1. Reference Model (Deep Copy & Freeze)
        import copy
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Dataset Handling
        ds = dataset if dataset is not None else getattr(model, 'dataset', None)
        if ds is None:
            self.logger.warning("Dataset not found in model. Ensure it is passed or available.")
            
        # Initialize Reward Calculator
        if ds:
            self.reward_calc = DeltaRewardCalculator(config, ds)
        else:
            self.reward_calc = None

        # 2. Config
        self.group_size = config.get('group_size', 8) # Recommend 8 for diversity
        # Alpha is deprecated in favor of Lexicographical Hierarchy
        self.beta = config.get('kl_beta', 0.1)        # DPO Constraint
        
        topk = config.get('topk', 10)
        self.k = topk[0] if isinstance(topk, list) else topk
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        
    def _create_reference_model(self, config, model):
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
        
    def _sample_lists(self, model, interaction, group_size, k):
        # 1. Get Logits
        scores = model.full_sort_predict(interaction)
        # Probabilities
        probs = F.softmax(scores, dim=-1)
        
        batch_size = probs.size(0)
        
        # 2. Expand: [Batch * G, Items]
        probs_expanded = probs.repeat_interleave(group_size, dim=0)
        
        # 3. Sample: [Batch * G, K]
        # Safety: Add epsilon to avoid error on zero-prob items
        probs_expanded = probs_expanded + 1e-8
        actions_flat = torch.multinomial(probs_expanded, num_samples=k, replacement=False)
        
        # 4. Compute Log Probs of SELECTED items
        # Gather probabilities of the sampled items
        # gathered_probs: [Batch * G, K]
        gathered_probs = torch.gather(probs_expanded, 1, actions_flat)
        
        # Sum log probs for the list (Sequence probability assumption)
        selected_log_probs = torch.log(gathered_probs + 1e-10).sum(dim=1)
        
        # 5. Reshape
        actions = actions_flat.view(batch_size, group_size, k)
        log_probs = selected_log_probs.view(batch_size, group_size)
        
        return actions, log_probs

    def _get_log_probs(self, model, interaction, actions):
        """
        Computes log probabilities of the given 'actions' (item lists).
        Output: [Batch] (Sum of log_probs of items in the list)
        """
        # actions: [Batch, K]
        # RecBole full_sort_predict returns [Batch, Num_Items]
        scores = model.full_sort_predict(interaction) 
        log_probs_all = F.log_softmax(scores, dim=-1)
        
        # Gather log_probs for the specific items in 'actions'
        # actions: [Batch, K]
        # gathered: [Batch, K]
        gathered_log_probs = torch.gather(log_probs_all, -1, actions)
        
        # Sum over K (Joint probability of the sequence)
        return gathered_log_probs.sum(dim=-1)

    def _predict_topk(self, model, interaction, k):
        """Greedy Top-K for Baseline."""
        scores = model.full_sort_predict(interaction)
        _, topk_items = torch.topk(scores, k, dim=-1)
        return topk_items.unsqueeze(1) # [Batch, 1, K]

    def get_preference_batch(self, batch_data, group_size=4):
        with torch.no_grad():
            # 1. Sample Lists (Student + Reference)
            student_lists, _ = self._sample_lists(self.model, batch_data, group_size, self.k)
            ref_list = self._predict_topk(self.ref_model, batch_data, self.k)
            
            # Pool: [Batch, G+1, K]
            # ref_list comes as [Batch, 1, K] from _predict_topk, need to verify
            # Looking at previous errors, _predict_topk returns [Batch, 1, K].
            # student_lists is [Batch, G, K].
            # So cat is safe on dim=1.
            candidate_pool = torch.cat([student_lists, ref_list], dim=1)
            
            # 2. Raw Metrics
            target_items = batch_data[self.config['ITEM_ID_FIELD']]
            all_ndcg = self.reward_calc.calc_ndcg(candidate_pool, target_items)
            all_pop = self.reward_calc.get_batch_pop(candidate_pool)
            
            # 3. Create Lexicographical Score (No Alpha)
            # We want strict hierarchy: NDCG > Serendipity
            
            # A. Normalize Pop (Unused in Pure NDCG)
            # pop_score = 1.0 / (all_pop + 1.0)
            
            # B. Pure NDCG Score
            # We ignore Serendipity.
            final_scores = all_ndcg
            
            # Safety: If NDCG is 0, we can drop the score to -1 to ensure it loses 
            # unless everyone is 0.
            final_scores = torch.where(all_ndcg > 0, final_scores, torch.tensor(-1.0, device=self.device))
            
            # 4. Select Winner/Loser
            best_indices = torch.argmax(final_scores, dim=1)
            worst_indices = torch.argmin(final_scores, dim=1)
            
            # Gather IDs
            batch_indices = torch.arange(candidate_pool.size(0), device=self.device)
            chosen_ids = candidate_pool[batch_indices, best_indices]
            rejected_ids = candidate_pool[batch_indices, worst_indices]
            
            # --- 5. The "Hierarchy" Mask ---
            
            # Get values for Winner
            chosen_ndcg = all_ndcg[batch_indices, best_indices]
            chosen_pop = all_pop[batch_indices, best_indices]
            
            # Get values for Loser
            rejected_ndcg = all_ndcg[batch_indices, worst_indices]
            rejected_pop = all_pop[batch_indices, worst_indices]
            
            # Condition A: NDCG Improvement (Primary Goal)
            # If Winner beats Loser in Accuracy, we ALWAYS train.
            is_ndcg_improved = (chosen_ndcg > rejected_ndcg)
            
            # Pure NDCG Improvement Mask
            # We ignore serendipity gains. Only strict accuracy gains count.
            valid_mask = is_ndcg_improved
            
            return chosen_ids, rejected_ids, valid_mask

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0.0
        num_valid_batches = 0
        
        iter_data = tqdm(train_data, desc=f"MODPO Epoch {epoch_idx}") if show_progress else train_data

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # 1. Get Preferences
            chosen_ids, rejected_ids, valid_mask = self.get_preference_batch(interaction, self.group_size)
            
            if valid_mask.sum() == 0:
                continue
                
            # Filter Batch
            # NOTE: RecBole Interaction supports slicing!
            valid_interaction = interaction[valid_mask]
            chosen_ids = chosen_ids[valid_mask]
            rejected_ids = rejected_ids[valid_mask]
            
            # 2. Policy Log Probs (Student)
            # _get_log_probs now returns [Batch], matching standard DPO
            policy_chosen_logps = self._get_log_probs(self.model, valid_interaction, chosen_ids)
            policy_rejected_logps = self._get_log_probs(self.model, valid_interaction, rejected_ids)
            
            # 3. Ref Log Probs (Teacher) - NO GRAD
            with torch.no_grad():
                ref_chosen_logps = self._get_log_probs(self.ref_model, valid_interaction, chosen_ids)
                ref_rejected_logps = self._get_log_probs(self.ref_model, valid_interaction, rejected_ids)

            # 4. DPO Loss Calculation
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            # The 'logits' for the sigmoid
            logits = self.beta * (policy_logratios - ref_logratios)
            
            # We want to maximize log_sigmoid(logits) -> minimize -log_sigmoid(logits)
            losses = -F.logsigmoid(logits)
            loss = losses.mean()
            
            loss.backward()
            if self.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # Logging
            if self.config.get('log_wandb', False):
                 import wandb
                 wandb.log({
                     'train/modpo_loss': loss.item(),
                     'train/dpo_accuracy': (logits > 0).float().mean().item(),
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
