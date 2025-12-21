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
        # --- 1. Calculate Deltas ---
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
        
        stud_pop = self.get_batch_pop(student_lists)
        base_pop = self.get_batch_pop(baseline_lists)

        # Accuracy Delta: Range [-1.0, 1.0]
        # Negative if Student < Baseline
        delta_ndcg = (stud_ndcg - base_ndcg)
        
        # Popularity Delta (Percentage): Range [-inf, 1.0]
        # Positive if Less Popular (Good). Negative if More Popular (Bad).
        delta_pop_pct = (base_pop - stud_pop) / (base_pop + 1e-9)

        # --- 2. Hierarchical Logic ---
        
        # A. The Gate: Did we maintain accuracy?
        # We allow a tiny margin of error (-0.001) to keep gradients flowing near the boundary
        pass_gate = (delta_ndcg >= -0.001)
        
        # --- 3. Reward Calculation ---
        
        # Term 1: Accuracy (Symmetric)
        # If delta_ndcg is negative, this is a PENALTY. 
        # This solves "ensure don't hurt base model performance".
        acc_term = alpha * delta_ndcg
        
        # Term 2: Serendipity (Conditional)
        # We only consider serendipity if the accuracy is acceptable.
        # We do NOT clamp min=0. If the model becomes more popular (delta_pop_pct < 0),
        # this term becomes negative, reducing the total reward.
        # This solves the "Pop 1639" explosion.
        pop_term = (1 - alpha) * delta_pop_pct
        
        # Apply Logic:
        # If Pass: Reward = Acc + Pop (Trade-off allowed)
        # If Fail: Reward = Acc only (Pure Penalty for failing accuracy)
        total_reward = torch.where(
            pass_gate,
            acc_term + pop_term,
            acc_term
        )
        
        # --- 4. Safety ---
        # Clamp to prevent gradient explosions from rare outliers
        total_reward = torch.clamp(total_reward, -1.0, 1.0)
        
        # Success Rate: Strictly beating baseline accuracy
        success_rate = (delta_ndcg > 0).float().mean()
        
        return total_reward, success_rate
