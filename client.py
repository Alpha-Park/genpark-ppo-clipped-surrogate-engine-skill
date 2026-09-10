class PPOClippedSurrogate:
    """
    Proximal Policy Optimization (PPO) Clipped Objective (Schulman et al.).
    Computes ratio r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s).
    L_CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t).
    """
    def __init__(self, epsilon=0.2):
        self.eps = epsilon

    def compute_loss(self, old_probs, new_probs, advantages):
        losses = []
        for p_old, p_new, adv in zip(old_probs, new_probs, advantages):
            ratio = p_new / (p_old + 1e-10)
            surr1 = ratio * adv
            clipped_ratio = max(1.0 - self.eps, min(1.0 + self.eps, ratio))
            surr2 = clipped_ratio * adv
            losses.append(min(surr1, surr2))
        return sum(losses) / len(losses) if losses else 0.0
