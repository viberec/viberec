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
        # --- 1. Metrics ---
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
        
        stud_pop = self.get_batch_pop(student_lists)
        base_pop = self.get_batch_pop(baseline_lists)

        # --- 2. Deltas ---
        delta_ndcg = (stud_ndcg - base_ndcg)
        # Pop Delta (Percentage): Positive = Good (Less Popular)
        delta_pop_pct = (base_pop - stud_pop) / (base_pop + 1e-9)

        # --- 3. The New Accuracy Logic (Solves the "Zero-Safe" Trap) ---
        # We start with the ABSOLUTE score (stud_ndcg).
        # Then we subtract the failure margin if we lost to the teacher.
        # Logic: Reward = stud_ndcg + min(0, delta_ndcg)
        # If Stud=0.4, Base=0.5 -> R = 0.4 - 0.1 = 0.3 (Positive! Better than 0)
        # If Stud=0.0, Base=0.0 -> R = 0.0 + 0.0 = 0.0
        
        # We apply alpha here to balance it with serendipity later
        acc_base = stud_ndcg
        acc_penalty = torch.clamp(delta_ndcg, max=0.0) 
        
        # The Accuracy Term:
        # We double-count the penalty slightly to keep the "Anchor" strong,
        # but the base `stud_ndcg` ensures we never prefer 0 over a decent try.
        acc_term = alpha * (acc_base + acc_penalty)
        
        # --- 4. The Serendipity Logic (Double Gated) ---
        # You only get serendipity if:
        # 1. You found the item (stud_ndcg > 0)
        # 2. You didn't lose significant accuracy (delta_ndcg >= -0.05)
        # We allow a tiny slack (-0.05) because acc_term is already punishing the drop.
        
        is_relevant = (stud_ndcg > 0)
        is_not_terrible = (delta_ndcg >= -0.05)
        gate_open = is_relevant & is_not_terrible
        
        pop_term = (1 - alpha) * delta_pop_pct * gate_open.float()
        
        # --- 5. Total Reward ---
        total_reward = acc_term + pop_term
        
        # Clip for stability
        total_reward = torch.clamp(total_reward, -1.0, 1.0)
        
        return total_reward, is_relevant.float().mean()
