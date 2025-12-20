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
        # --- 1. Calculate Raw Metrics ---
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        base_ndcg = self.calc_ndcg(baseline_lists, ground_truth) # [Batch, 1]
        
        stud_pop = self.get_batch_pop(student_lists)
        base_pop = self.get_batch_pop(baseline_lists) # [Batch, 1]

        # --- 2. Calculate Deltas (FIXED UNITS) ---
        
        # Accuracy Delta: Range [-1.0, 1.0]
        delta_ndcg = (stud_ndcg - base_ndcg)
        
        # Popularity Delta: Convert to PERCENTAGE IMPROVEMENT
        # Formula: (Base - Student) / Base
        # Range: usually [0.0, 1.0] (if less popular). 
        # Can be negative if student is MORE popular.
        delta_pop_pct = (base_pop - stud_pop) / (base_pop + 1e-9)

        # --- 3. Hierarchical Logic (FIXED GATE) ---
        
        # A. The Gate: 
        # Condition 1: Must not be worse than baseline (delta_ndcg >= 0)
        # Condition 2: MUST BE RELEVANT (stud_ndcg > 0) <--- CRITICAL FIX
        # This prevents the "0 >= 0" loophole where garbage gets rewarded.
        gate_open = (delta_ndcg >= 0) & (stud_ndcg > 0)
        accuracy_gate = gate_open.float()
        
        # B. The Bonus (Serendipity)
        # "No Penalty" -> Clamp min=0
        # "Normalized" -> Use delta_pop_pct
        pop_bonus = torch.clamp(delta_pop_pct, min=0.0)
        
        # Scale bonus to be comparable to accuracy
        # Since pop_pct is [0,1] and ndcg is [0,1], we can trust alpha now.
        weighted_bonus = (1 - alpha) * pop_bonus * accuracy_gate
        
        # --- 4. Total Reward ---
        # Base: Accuracy Gradient (Always Active)
        base_reward = alpha * delta_ndcg
        
        total_reward = base_reward + weighted_bonus
        
        # --- 5. Safety Clamp ---
        # Keep reward in [-1, 1] for stable gradients
        total_reward = torch.clamp(total_reward, -1.0, 1.0)
        
        # Success Rate: How often did we get the bonus?
        success_rate = accuracy_gate.mean()
        
        return total_reward, success_rate
