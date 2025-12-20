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
        Computes reward based on the RANK of the Delta Improvement within the group.
        Range: [0.0, 1.0]
        """
        batch_size, group_size, _ = student_lists.shape
        
        # --- 1. Calculate Raw Metrics ---
        # [Batch, Group]
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        stud_pop  = self.get_batch_pop(student_lists)
        
        with torch.no_grad():
            # [Batch, 1]
            base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
            base_pop  = self.get_batch_pop(baseline_lists)

        # --- 2. Calculate Deltas ---
        # We want to rank "How much did we IMPROVE?"
        # Positive Delta = Good. Negative Delta = Bad.
        
        # Accuracy Delta: Higher is Better
        delta_ndcg = (stud_ndcg - base_ndcg)
        
        # Serendipity Delta: (Base - Stud). Higher is Better (Lower Student Pop)
        delta_pop = (base_pop - stud_pop)
        
        # --- 3. Compute Ranks (The Magic Step) ---
        # argsort().argsort() returns the rank index (0 to G-1)
        # We apply this along the Group dimension (dim=1)
        
        # Rank 0 = Worst improvement (or biggest drop)
        # Rank G-1 = Best improvement
        rank_ndcg = delta_ndcg.argsort(dim=1).argsort(dim=1).float()
        rank_pop  = delta_pop.argsort(dim=1).argsort(dim=1).float()
        
        # --- 4. Normalize Ranks to [0, 1] ---
        # This makes the reward Unit-Free.
        # We add 1e-8 to denominator to avoid div-by-zero if group_size=1
        norm_rank_ndcg = rank_ndcg / (group_size - 1 + 1e-8)
        norm_rank_pop  = rank_pop  / (group_size - 1 + 1e-8)
        
        # --- 5. Weighted Sum ---
        # Since both are [0,1], alpha works strictly as a percentage mixer.
        total_reward = (alpha * norm_rank_ndcg) + ((1 - alpha) * norm_rank_pop)
        
        # --- 6. The "Tie-Breaker" (Optional but Recommended) ---
        # If multiple items have identical NDCG (e.g., all 0), their ranks might be arbitrary.
        # But usually argsort handles stability.
        # We can rely on the fact that total_reward is now strictly bounded [0, 1].
        
        # Success Rate (Did we actually beat the baseline absolute score?)
        success_rate = (delta_ndcg >= 0).float().mean()
        
        return total_reward, success_rate
