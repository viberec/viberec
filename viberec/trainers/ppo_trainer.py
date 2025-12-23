from recbole.trainer import Trainer
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import copy
import wandb
from viberec.trainers.rl_trainer import RLFinetuneTrainer

class PPOTrainer(RLFinetuneTrainer):
    """
    PPO-Clip Trainer for Sequential Recommendation.
    Architecture: Actor-Only (Single Network).
    Advantage Estimation: Moving Average Baseline (Variance Reduction).
    """
    def __init__(self, config, model, dataset=None):
        super(PPOTrainer, self).__init__(config, model)
        
        # PPO Hyperparameters
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.target_kl = config.get('target_kl', 0.01)  # NEW: Early stopping threshold
        
        # Stability: Moving Average for Advantage Baseline
        self.running_reward_mean = 0.0
        self.running_reward_std = 1.0
        self.momentum = 0.9  # For updating the moving average

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        iter_data = tqdm(train_data, desc=f"PPO Epoch {epoch_idx}") if show_progress else train_data

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            target_items = interaction[self.config['ITEM_ID_FIELD']]
            
            # --- 1. Sampling (On-Policy Experience) ---
            with torch.no_grad():
                # Force group_size=1 for PPO (Strict On-Policy)
                # actions: [Batch, 1, K] or [Batch, K]
                actions, old_log_probs = self._sample_lists(self.model, interaction, self.k, group_size=1)
                
                # Rewards (NDCG)
                # calc_ndcg needs [Batch, Group=1, K] or [Batch, K]
                # It returns [Batch, Group] or [Batch, 1] usually.
                # If actions is [Batch, K], it's treated as Group=1
                
                rewards = self.calc_ndcg(actions, target_items)
                if rewards.ndim > 1: rewards = rewards.squeeze(1) # Flatten to [Batch]
                
                # --- Update 2: Moving Average Advantage ---
                # This prevents "Good Batches" from being zeroed out
                batch_mean = rewards.mean().item()
                batch_std = rewards.std().item()
                
                # Update global stats
                if num_batches == 0 and epoch_idx == 0:
                    self.running_reward_mean = batch_mean
                    self.running_reward_std = batch_std if batch_std > 0 else 1.0
                else:
                    self.running_reward_mean = self.momentum * self.running_reward_mean + (1 - self.momentum) * batch_mean
                    self.running_reward_std = self.momentum * self.running_reward_std + (1 - self.momentum) * batch_std
                
                # Calculate Advantage using GLOBAL stats (Critic Proxy)
                # We normalize strictly to keep gradients stable
                advantages = (rewards - self.running_reward_mean) / (self.running_reward_std + 1e-8)
            
            # --- 2. PPO Update Loop ---
            for _ in range(self.ppo_epochs):
                self.optimizer.zero_grad()
                
                # New Probabilities
                new_log_probs, all_log_probs_dist = self._get_log_probs(self.model, interaction, actions)
                if new_log_probs.ndim > 1: new_log_probs = new_log_probs.squeeze(1)
                
                # Ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Surrogate Loss (PPO-Clip)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy Loss
                probs = torch.exp(all_log_probs_dist)
                entropy = -(probs * all_log_probs_dist).sum(dim=-1).mean()
                
                # Total Loss
                loss = policy_loss - self.entropy_coef * entropy
                loss.backward()
                
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                self.optimizer.step()
                
                # --- Update 3: KL Divergence Check ---
                # Calculate approx KL to see if we drifted too far
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                
                if approx_kl > 1.5 * self.target_kl:
                    break # Early stop this batch to prevent collapse
                
                total_loss += loss.item()
            
            # Logging (Last inner step)
            if self.config.get('log_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
                    'train/reward_mean': batch_mean, 
                    'train/entropy': entropy.item(),
                    'epoch': epoch_idx
                })

            num_batches += 1

        avg_loss = total_loss / (num_batches * self.ppo_epochs) if num_batches > 0 else 0.0
        self.logger.info(f"Epoch {epoch_idx} done. Loss: {avg_loss:.4f}")
        return avg_loss
