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

        # --- 2. Raw Deltas ---
        # Accuracy Delta (Small scale: ~0.0 to 1.0)
        delta_ndcg = (stud_ndcg - base_ndcg)
        
        # Popularity Delta (Huge scale: ~0 to 1000+)
        # We use raw counts because Scale Matching handles the unit conversion.
        # Positive = Improvement (Less Popular)
        delta_pop = (base_pop - stud_pop) 

        # --- 3. Dynamic Scale Matching (No Magic Numbers) ---
        # We calculate the Standard Deviation of both metrics across the entire batch.
        # This tells us the "typical" fluctuation magnitude for this training step.
        
        std_ndcg = torch.std(delta_ndcg)
        std_pop  = torch.std(delta_pop)
        
        # Calculate the Ratio: "How much bigger is Pop than NDCG?"
        # If Pop varies by 500 and NDCG varies by 0.1, ratio is ~0.0002
        # We use detach() because we don't want to backprop through the std calculation.
        scale_ratio = (std_ndcg / (std_pop + 1e-9)).detach()
        
        # Normalize Popularity to match Accuracy's physics
        # Now, a "big" jump in popularity is mathematically equal to a "big" jump in accuracy.
        normalized_pop = delta_pop * scale_ratio
        
        # --- 4. Total Reward Calculation ---
        
        # Term 1: Accuracy (Weighted by Alpha)
        # We reinforce the absolute hit (stud_ndcg) and the relative gain (delta_ndcg)
        # to prevent the "Zero-Safe" trap.
        acc_term = alpha * (stud_ndcg + delta_ndcg)
        
        # Term 2: Serendipity (Weighted by 1-Alpha)
        # We use the Normalized Pop, so we don't need magic damping factors.
        pop_term = (1 - alpha) * normalized_pop
        
        # --- 5. The "Relevance Gate" ---
        # Even with scaling, we only want to play the serendipity game 
        # if the item is relevant.
        is_relevant = (stud_ndcg > 0).float()
        
        # Final Sum:
        # If irrelevant: Reward = 0 (Total failure, prevents exploitation of garbage)
        # If relevant: Reward = Acc + Normalized_Pop
        total_reward = (acc_term + pop_term) * is_relevant
        
        # Safety Clamp (Just to keep PPO stable)
        total_reward = torch.clamp(total_reward, -1.0, 1.0)
        
        return total_reward, is_relevant.mean()

class DPOReferenceRewardCalculator(DeltaRewardCalculator):
    """
    Reward logic specifically adapted for DPO.
    In DPO, we don't need a scalar reward for gradient ascent directly in the loss.
    Instead, we need a scalar 'utility' score to determine which sample is 'Chosen' vs 'Rejected'.
    
    This utility score should reflect the user's preference:
    Prefer accuracy + serendipity > accuracy > pure serendipity > infinite loop.
    """
    def compute_reward(self, student_lists, baseline_lists, ground_truth, alpha):
        # reuse parent metrics calculation
        stud_ndcg = self.calc_ndcg(student_lists, ground_truth)
        stud_pop = self.get_batch_pop(student_lists)
        
        # We don't necessarily need baseline_lists for DPO *ranking* logic if we just want absolute quality,
        # BUT if we want "improvement over baseline" to be the utility key, we keep it.
        # Let's stick to the "improvement" logic to be consistent with GRPO objective.
        
        base_ndcg = self.calc_ndcg(baseline_lists, ground_truth)
        base_pop = self.get_batch_pop(baseline_lists)
        
        # --- Utility Score Calculation ---
        # 1. Accuracy Utility
        # High NDCG = Good.
        # Improvement over Baseline = Very Good.
        delta_ndcg = stud_ndcg - base_ndcg
        
        # 2. Serendipity Utility
        # Low Pop = Good.
        delta_pop_pct = (base_pop - stud_pop) / (base_pop + 1e-9)
        # Squash pop signal
        pop_signal = torch.tanh(delta_pop_pct)
        
        # 3. Combine
        # "Alpha" controls importance of Accuracy.
        # Utility = Alpha * Accuracy_Signal + (1-Alpha) * Serendipity_Signal
        
        # We use a simpler additive form for Preference Ranking (Values don't need to be bounded strictly)
        # However, accurate ranking is critical.
        
        # Gating: If item is irrelevant (NDCG=0), Utility should be very low (e.g. -infinity or just 0).
        # We want to prefer Relevant items over everything else.
        
        is_relevant = (stud_ndcg > 0).float()
        
        # Accuracy Part:
        # We reward high absolute NDCG AND improvement.
        acc_utility = stud_ndcg + delta_ndcg
        
        # Serendipity Part:
        # Only counts if relevant.
        pop_utility = pop_signal * is_relevant
        
        total_utility = (alpha * acc_utility) + ((1 - alpha) * pop_utility)
        
        # Success Rate (for logging only)
        success_rate = is_relevant.mean()
        
        return total_utility, success_rate
