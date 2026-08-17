
import numpy as np

class CausalRewardShaper:
    """Potential-based reward shaping using dynamic structural causal weights."""
    
    def __init__(self, weights=None, alpha=1.0, beta=3.0):
        self.alpha = alpha
        self.beta = beta
        self.weights = np.array(weights) if weights is not None else np.array([0.12, 0.10, 0.52, 0.30, 0.15, 0.40])

    def compute_potential(self, state_vector):
        """Computes scalar potential Phi(s) = w^T * s."""
        state = np.array(state_vector, dtype=float)
        return float(np.dot(self.weights, state))

    def compute_shaped_reward(self, current_state, previous_state):
        """Calculates R_SCM = alpha * Phi(s_t) + beta * (Phi(s_t) - Phi(s_{t-1}))."""
        phi_curr = self.compute_potential(current_state)
        phi_prev = self.compute_potential(previous_state)
        return self.alpha * phi_curr + self.beta * (phi_curr - phi_prev)
