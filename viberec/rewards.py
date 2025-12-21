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
        # alpha is now the "Amplification Factor" (e.g., 0.5)
        # It controls how much popularity can boost/nerf the score.
        
        # --- 1. The Foundation (Sparse Signal) ---
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        # Range: [0.0, 1.0]
        
        # --- 2. The Modulator (Dense Signal) ---
        stud_pop = self.get_batch_pop(student_lists)
        
        with torch.no_grad():
            base_pop = self.get_batch_pop(baseline_lists)

        # Calculate Percentage Change
        # Positive = Improved Serendipity (Less Popular)
        # Negative = More Popular
        delta_pop_pct = (base_pop - stud_pop) / (base_pop + 1e-9)
        
        # --- 3. Scale Fix: Tanh Saturation ---
        # This is the magic step.
        # tanh(0.1) ≈ 0.1, but tanh(5.0) ≈ 1.0.
        # It linearly rewards small improvements but aggressively caps outliers.
        # The scale is now strictly [-1.0, 1.0].
        pop_signal = torch.tanh(delta_pop_pct)
        
        # --- 4. The Multiplicative Logic ---
        # Formula: Reward = NDCG * (1 + alpha * PopSignal)
        
        # Examples with alpha=0.5:
        # A. Hit Item (1.0), Neutral Pop (0.0) -> R = 1.0 * (1 + 0) = 1.0
        # B. Hit Item (1.0), Great Pop (+1.0)  -> R = 1.0 * (1 + 0.5) = 1.5 (Bonus!)
        # C. Hit Item (1.0), Bad Pop (-1.0)    -> R = 1.0 * (1 - 0.5) = 0.5 (Dampened, but NOT Negative)
        # D. Miss Item (0.0), Great Pop (+1.0) -> R = 0.0 * (1.5) = 0.0 (Garbage filtered)
        
        modulation = 1.0 + ((1 - alpha) * pop_signal)
        
        # Ensure modulation doesn't flip sign (if alpha > 1, bad pop could make reward negative)
        # We clip modulation to [0.1, 2.0] to be safe.
        modulation = torch.clamp(modulation, min=0.1, max=2.0)
        
        total_reward = stud_ndcg * modulation
        
        # --- 5. Success Rate ---
        # Strictly finding the item
        success_rate = (stud_ndcg > 0).float().mean()
        
        return total_reward, success_rate
