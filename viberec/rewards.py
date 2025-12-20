import torch

class DeltaRewardCalculator:
    def __init__(self, config, dataset):
        """
        Initializes the reward calculator with item popularity stats.
        
        Args:
            config: RecBole Config object (needs 'device' and 'topk').
            dataset: RecBole Dataset object (to calculate item frequency).
        """
        self.device = config['device']
        
        # Handle topk: RecBole config usually stores it as a list/int
        if isinstance(config['topk'], list):
            self.k = config['topk'][0]  # e.g., 10
        else:
            self.k = int(config['topk'])
        
        # 1. Pre-compute Item Popularity (Raw Counts)
        # We use a tensor of shape [item_num] where index = item_id, value = count
        self.item_pop = torch.zeros(dataset.item_num, device=self.device)
        
        # Get counts from the dataset
        # Adapted for RecBole Dataset which provides item_counter
        for iid, count in dataset.item_counter.items():
            self.item_pop[iid] = float(count)

        # Pre-compute IDCG constants for NDCG speedup
        # 1 / log2(rank + 2)
        self.discounts = 1.0 / torch.log2(torch.arange(self.k, device=self.device).float() + 2.0)

    def get_batch_pop(self, item_lists):
        """
        Calculates the Average Popularity for a batch of lists.
        
        Args:
            item_lists: Tensor [Batch, Group, K] (Item Indices)
        Returns:
            pop_scores: Tensor [Batch, Group]
        """
        # Look up popularity counts: [Batch, Group, K]
        pop_counts = self.item_pop[item_lists] 
        
        # Mean over the K items in the list: [Batch, Group]
        return pop_counts.mean(dim=-1)

    def calc_ndcg(self, recommended_lists, ground_truth):
        """
        Calculates NDCG@K for RL batches.
        
        Args:
            recommended_lists: Tensor [Batch, Group, K]
            ground_truth: Tensor [Batch] (Target Item IDs)
            
        Returns:
            ndcg_scores: Tensor [Batch, Group]
        """
        # Handle shape mismatch if Group dimension is missing (e.g. baseline might be [B, K])
        if recommended_lists.ndim == 2:
            recommended_lists = recommended_lists.unsqueeze(1) # [B, 1, K]
            
        batch_size, group_size, k = recommended_lists.shape
        
        # 1. Expand Ground Truth for comparison
        # Shape: [Batch, 1, 1] -> broadcast against [Batch, Group, K]
        gt_expanded = ground_truth.view(batch_size, 1, 1)
        
        # 2. Check for Hits (Binary Mask)
        # hits shape: [Batch, Group, K]
        hits = (recommended_lists == gt_expanded).float()
        
        # 3. Compute DCG
        # Broadcast discounts [K] against hits [Batch, Group, K]
        dcg = (hits * self.discounts).sum(dim=-1) # Sum over K -> [Batch, Group]
        
        # 4. Compute IDCG 
        # Since we usually have 1 ground truth item per user in Next-Item prediction,
        # IDCG is always 1.0 (if the item is in the top K positions).
        # Standard RecBole NDCG simplifies to DCG / 1.0 for Leave-One-Out.
        idcg = 1.0
        
        return dcg / idcg

    def compute_reward(self, student_lists, baseline_lists, ground_truth, alpha):
        """
        Computes the Mixed Relative Reward.
        
        Args:
            student_lists:  Tensor [Batch, Group, K] (Sampled from Policy)
            baseline_lists: Tensor [Batch, 1, K] (Greedy output from Base Model)
            ground_truth:   Tensor [Batch]
            alpha:          Float (0.0 to 1.0), weight for Accuracy.
            
        Returns:
            total_reward: Tensor [Batch, Group]
        """
        # Ensure input shapes are consistent for broadcasting
        if student_lists.ndim == 2:
            student_lists = student_lists.unsqueeze(1)
        if baseline_lists.ndim == 2:
            baseline_lists = baseline_lists.unsqueeze(1)
            
        # --- 1. Calculate Metrics ---
        # Student Metrics: [Batch, Group]
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        stud_pop  = self.get_batch_pop(student_lists)
        
        # Baseline Metrics: [Batch, 1] (No gradients needed for baseline values)
        with torch.no_grad():
            base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
            base_pop  = self.get_batch_pop(baseline_lists)
            
        # --- 2. Calculate Deltas (Relative Improvement) ---
        
        # Delta NDCG: (Student - Base) / Base
        # Higher student NDCG is better.
        # Add epsilon 1e-8 to avoid division by zero.
        delta_ndcg = (stud_ndcg - base_ndcg) / (base_ndcg + 1e-8)
        
        # Delta Pop: (Base - Student) / Base
        # Lower student Popularity is better (Higher Serendipity).
        # If Student < Base, result is Positive.
        delta_pop = (base_pop - stud_pop) / (base_pop + 1e-8)
        
        # --- 3. Stability Clipping (Crucial for RL) ---
        # Prevents gradients from exploding if baseline is tiny.
        # Range [-1.0, 1.0] usually works well for PPO/GRPO.
        delta_ndcg = torch.clamp(delta_ndcg, min=-1.0, max=1.0)
        delta_pop  = torch.clamp(delta_pop, min=-1.0, max=1.0)
        
        # --- 3. Weighted Full Reward (Before Floor) ---
        full_reward = (alpha * delta_ndcg) + ((1 - alpha) * delta_pop)
        
        # --- 4. The Safety Floor ---
        # Logic: 
        # If delta_ndcg < 0: Penalty = delta_ndcg (Negative) -> Signal to get back to baseline
        # If delta_ndcg >= 0: Reward = full_reward (Weighted trade-off)
        total_reward = torch.where(
            (delta_ndcg >= 0) & (delta_pop >= 0), 
            full_reward, 
            delta_ndcg 
        )
        
        # Calculate "Success Rate" for logging (How often we match/beat baseline)
        success_rate = ((delta_ndcg >= 0) & (delta_pop >= 0)).float().mean()
        
        return torch.clamp(total_reward, -1.0, 1.0), success_rate
