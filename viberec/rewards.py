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
        # 1. Calculate Raw Metrics [Batch, Group]
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        stud_pop  = self.get_batch_pop(student_lists)
        
        # 2. Compute Ranks within the Group (Batch-wise)
        # argsort twice gives us the rank (0 to G-1)
        # We want Higher NDCG -> Higher Rank
        rank_ndcg = stud_ndcg.argsort(dim=1).argsort(dim=1).float()
        
        # We want Lower Pop -> Higher Rank (so we negate stud_pop before sorting)
        rank_pop = (-stud_pop).argsort(dim=1).argsort(dim=1).float()
        
        # 3. Normalize Ranks to [0, 1] range
        # Divide by (Group_Size - 1)
        group_size = student_lists.size(1)
        norm_rank_ndcg = rank_ndcg / (group_size - 1 + 1e-8)
        norm_rank_pop  = rank_pop  / (group_size - 1 + 1e-8)
        
        # 4. Combine (Purely Alpha-controlled)
        # No magic numbers needed. alpha=0.5 means EXACTLY equal weight.
        total_reward = (alpha * norm_rank_ndcg) + ((1 - alpha) * norm_rank_pop)
        
        # 5. Success Rate (for logging)
        # We track how often the chosen winner beat the baseline
        with torch.no_grad():
            base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
            # Success if student matches/beats baseline NDCG
            is_success = (stud_ndcg >= base_ndcg)
            
        return total_reward, is_success.float().mean()
