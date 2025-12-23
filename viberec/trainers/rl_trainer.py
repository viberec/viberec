from recbole.trainer import Trainer
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy
import wandb
from tqdm import tqdm

class RLFinetuneTrainer(Trainer):
    """
    Base class for RL-based fine-tuning trainers (DPO, PPO).
    Provides common functionality for sampling, reward calculation (NDCG), and logging.
    """
    def __init__(self, config, model, dataset=None):
        super(RLFinetuneTrainer, self).__init__(config, model)
        
        # Common Config
        topk = config.get('topk', 10)
        self.k = topk[0] if isinstance(topk, list) else topk
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        
        # Pre-compute IDCG constants for NDCG speedup
        # 1 / log2(rank + 2)
        self.discounts = 1.0 / torch.log2(torch.arange(self.k, device=self.device).float() + 2.0)

    def calc_ndcg(self, recommended_lists, ground_truth):
        """
        Calculates NDCG@K for sampled lists.
        Compatible with 2D [Batch, K] and 3D [Batch, Group, K] inputs.
        """
        # Handle shape mismatch if Group dimension is missing (Standard PPO case)
        if recommended_lists.ndim == 2:
            recommended_lists = recommended_lists.unsqueeze(1) # [B, 1, K]
            
        batch_size, group_size, k = recommended_lists.shape
        
        # 1. Expand Ground Truth
        # Shape: [Batch, 1, 1] -> broadcast against [Batch, Group, K]
        gt_expanded = ground_truth.view(batch_size, 1, 1)
        
        # 2. Check Hits
        hits = (recommended_lists == gt_expanded).float()
        
        # 3. Compute DCG
        # Broadcast discounts [K] against hits [Batch, Group, K]
        dcg = (hits * self.discounts).sum(dim=-1) # Sum over K -> [Batch, Group]
        
        # 4. IDCG is 1.0 (since we only define success if target is in top K, and max relevance is 1)
        return dcg # [Batch, Group]

    def _sample_lists(self, model, interaction, k, group_size=1):
        """
        Sample K items based on policy probabilities.
        Returns:
            actions: [Batch, Group, K] (or [Batch, K] if group_size=1)
            sequence_log_probs: [Batch, Group] (or [Batch] if group_size=1)
        """
        scores = model.full_sort_predict(interaction)
        probs = F.softmax(scores, dim=-1)
        
        # Safety for multinomial
        probs = probs + 1e-8
        
        # Handle group size by repeating if necessary
        # Usually DPO repeats probs before sampling?
        # PPO usually samples once per batch item.
        # This common implementation assumes efficient multinomial sampling.
        
        if group_size > 1:
            # Expand probs: [Batch * G, Items]
            probs = probs.repeat_interleave(group_size, dim=0)
            
        # Sample: [Batch * G, K] or [Batch, K]
        actions_flat = torch.multinomial(probs, num_samples=k, replacement=False)
        
        # Compute Log Probs (re-compute from scores to be safe and cleaner graph)
        # Note: scores need expansion too if group_size > 1
        log_probs_all = F.log_softmax(scores, dim=-1)
        if group_size > 1:
            log_probs_all = log_probs_all.repeat_interleave(group_size, dim=0)
            
        action_log_probs = torch.gather(log_probs_all, 1, actions_flat)
        sequence_log_probs_flat = action_log_probs.sum(dim=-1)
        
        if group_size > 1:
            batch_size = interaction[self.config['USER_ID_FIELD']].size(0)
            actions = actions_flat.view(batch_size, group_size, k)
            sequence_log_probs = sequence_log_probs_flat.view(batch_size, group_size)
        else:
            actions = actions_flat
            sequence_log_probs = sequence_log_probs_flat
            
        return actions, sequence_log_probs

    def _get_log_probs(self, model, interaction, actions):
        """
        Compute log probs of already sampled actions under current model state.
        Args:
            actions: [Batch, Group, K] or [Batch, K]
        Returns:
            sum_log_probs: [Batch, Group] or [Batch]
            all_log_probs: [Batch, Group, Items] or [Batch, Items] (Distribution)
        """
        scores = model.full_sort_predict(interaction)
        log_probs_all = F.log_softmax(scores, dim=-1)
        
        # Handle dimensions
        is_grouped = (actions.ndim == 3)
        if is_grouped:
            group_size = actions.size(1)
            # Expand log_probs: [Batch * G, Items]
            log_probs_expanded = log_probs_all.repeat_interleave(group_size, dim=0)
            actions_flat = actions.view(-1, actions.size(-1))
            
            gathered = torch.gather(log_probs_expanded, 1, actions_flat)
            sum_log_probs = gathered.sum(dim=-1).view(actions.size(0), group_size)
            
            return sum_log_probs, log_probs_all # Return original distribution (unexpanded) usually? 
            # DPO doesn't need distribution. PPO needs distribution for Entropy.
            # PPO usually not grouped.
        else:
            gathered = torch.gather(log_probs_all, 1, actions)
            sum_log_probs = gathered.sum(dim=-1)
            return sum_log_probs, log_probs_all
            
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        raise NotImplementedError
