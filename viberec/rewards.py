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
        Hierarchical Reward:
        1. Base = Delta NDCG.
        2. Bonus = Delta Pop (Only if NDCG >= Baseline, and ONLY if Pop improved).
        """
        # --- 1. Calculate Deltas ---
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        base_ndcg = self.calc_ndcg(baseline_lists, ground_truth) # No grad needed usually, but safe to keep
        
        stud_pop = self.get_batch_pop(student_lists)
        base_pop = self.get_batch_pop(baseline_lists)

        # Relative Changes
        delta_ndcg = (stud_ndcg - base_ndcg)
        delta_pop  = (base_pop - stud_pop) # Positive = Serendipity Improved

        # --- 2. Hierarchical Logic ---
        
        # A. The Gate: Did we maintain or improve accuracy?
        # We use a float mask: 1.0 if passed, 0.0 if failed
        accuracy_gate = (delta_ndcg >= 0).float()
        
        # B. The Bonus (Serendipity)
        # User Requirement 1: "Do not penalty" -> We clamp min=0. 
        # If delta_pop is negative (more popular), bonus is just 0. No punishment.
        pop_bonus = torch.clamp(delta_pop, min=0.0)
        
        # User Requirement 2: "Only focus delta pop if ndcg > 0"
        # We scale the bonus by alpha (or 1-alpha) to control magnitude relative to accuracy
        # If gate is closed (0), this entire term vanishes.
        weighted_bonus = (1 - alpha) * pop_bonus * accuracy_gate
        
        # --- 3. Total Reward ---
        # Case Fail:   delta_ndcg (Negative) + 0             -> Pure Accuracy Penalty
        # Case Pass:   delta_ndcg (Positive) + weighted_bonus -> Accuracy + Serendipity
        
        # Note: We weight delta_ndcg by 'alpha' to keep the scales comparable
        base_reward = alpha * delta_ndcg
        
        total_reward = base_reward + weighted_bonus
        
        # Calculate Success Rate (How often did we unlock the gate?)
        success_rate = accuracy_gate.mean()
        
        return total_reward, success_rate
