from recbole.trainer import Trainer
from recbole.utils import early_stopping
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any

from viberec.rewards import DeltaRewardCalculator

class GRPOTrainer(Trainer):
    """
    GRPOTrainer implements Group Relative Policy Optimization (GRPO) for fine-tuning.
    """
    def __init__(self, config, model, dataset=None):
        # Pass config and model to parent BEFORE initializing custom logic
        super(GRPOTrainer, self).__init__(config, model)
        
        # 1. Dataset Handling
        # Standard RecBole models usually have self.dataset. 
        # If passed explicitly, use it; otherwise fallback to model.dataset.
        ds = dataset if dataset is not None else getattr(model, 'dataset', None)
        if ds is None:
            # Fallback for some RecBole versions where dataset is in train_data
            # We will handle this in fit() or assume user provides it.
            self.logger.warning("Dataset not found in model. Ensure it is passed or available.")
            
        # Initialize Reward Calculator
        # Note: DeltaRewardCalculator needs 'ds' to calculate popularity.
        if ds:
            self.reward_calc = DeltaRewardCalculator(config, ds)
        else:
            self.reward_calc = None # Handle error later if still None
        
        # 2. Reference Model
        self.ref_model = self._create_reference_model(config, model)
        
        # 3. Hyperparameters
        self.group_size = config.get('group_size', 4)
        self.alpha = config.get('alpha', 0.5)
        # Map kl_beta to beta
        self.beta = config.get('kl_beta', 0.01) 
        
        # Handle TopK: RecBole often passes list like [10, 20], take first.
        topk = config.get('topk', 10)
        self.k = topk[0] if isinstance(topk, list) else topk
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        
        self.logger.info(f"[GRPO Trainer] Init: G={self.group_size}, Alpha={self.alpha}, KL_Beta={self.beta}, K={self.k}, Clip={self.clip_grad_norm}")
        
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
        # Safety: Add epsilon to avoid error on zero-prob items (though rare with softmax)
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
        """Compute log probs of EXISTING actions under a specific model."""
        scores = model.full_sort_predict(interaction)
        probs = F.softmax(scores, dim=-1) # [Batch, Items]
        
        batch_size, group_size, k = actions.shape
        
        # Reshaping
        actions_flat = actions.view(-1, k)
        probs_expanded = probs.repeat_interleave(group_size, dim=0)
        
        # Gather
        gathered_probs = torch.gather(probs_expanded, 1, actions_flat)
        selected_log_probs = torch.log(gathered_probs + 1e-10).sum(dim=1)
        
        return selected_log_probs.view(batch_size, group_size)

    def _predict_topk(self, model, interaction, k):
        """Greedy Top-K for Baseline."""
        scores = model.full_sort_predict(interaction)
        _, topk_items = torch.topk(scores, k, dim=-1)
        return topk_items.unsqueeze(1) # [Batch, 1, K]

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_success = 0.0
        total_kl = 0.0
        
        # TQDM handling
        iter_data = tqdm(train_data, desc=f"Epoch {epoch_idx}") if show_progress else train_data

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            # --- A. Student Sampling ---
            student_lists, student_log_probs = self._sample_lists(
                self.model, interaction, self.group_size, self.k
            )

            # --- B. Reference Calculations (No Grad) ---
            with torch.no_grad():
                # 1. KL Reference Log Probs
                ref_log_probs = self._get_log_probs(self.ref_model, interaction, student_lists)
                # 2. Baseline Lists for Delta Reward
                baseline_lists = self._predict_topk(self.ref_model, interaction, self.k)

            # --- C. Reward ---
            # Ensure Dataset is attached for popularity lookup
            target_items = interaction[self.config['ITEM_ID_FIELD']]
            
            # [Batch, Group]
            rewards, success_rate = self.reward_calc.compute_reward(
                student_lists, baseline_lists, target_items, alpha=self.alpha
            )
            total_success += success_rate.item()

            # --- D. Advantage ---
            mean_rew = rewards.mean(dim=1, keepdim=True)
            std_rew = rewards.std(dim=1, keepdim=True)
            advantages = (rewards - mean_rew) / (std_rew + 1e-8)

            # --- E. Loss ---
            # IMPORTANT: Detach advantages!
            # We treat advantage as a constant scaling factor for the gradient.
            advantages = advantages.detach()
            
            # KL Divergence (Approx): Log(Student) - Log(Ref)
            # We want to penalized if Student diverges from Ref
            kl_div = student_log_probs - ref_log_probs
            
            # Policy Gradient Loss: -1 * Advantage * LogProb
            pg_loss = -1 * (advantages * student_log_probs).mean()
            
            # KL Loss: Beta * KL
            kl_loss = self.beta * kl_div.mean()
            
            loss = pg_loss + kl_loss
            
            # --- F. Update ---
            loss.backward()
            if self.clip_grad_norm and self.clip_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # Logging
            if self.config.get('log_wandb', False):
                 import wandb
                 wandb.log({
                     'train/loss': loss.item(),
                     'train/reward': rewards.mean().item(),
                     'train/kl_div': kl_div.mean().item(),
                     'train/pg_loss': pg_loss.item(),
                     'epoch': epoch_idx
                 })

            total_loss += loss.item()
            total_reward += rewards.mean().item()
            total_kl += kl_div.mean().item()

        # Logging
        avg_loss = total_loss / len(train_data)
        avg_reward = total_reward / len(train_data)
        avg_success = total_success / len(train_data)
        self.logger.info(f"Epoch {epoch_idx} done. Loss: {avg_loss:.4f} | Rew: {avg_reward:.4f} | Success: {avg_success:.2%}")
        
        return avg_loss