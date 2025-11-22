import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        # Initialize action-value estimates Q(a) <- 0 
        self.q_values = np.zeros(n_arms) 
        # Initialize action counts N(a) <- 0 
        self.action_counts = np.zeros(n_arms)
        
        # True means (hidden from agent) for simulation purposes
        self.true_means = np.random.normal(0, 1, n_arms)

    def select_action(self):
        # Generate random number r 
        r = np.random.random()
        
        # If r < epsilon, Select random action (Explore) 
        if r < self.epsilon:
            return np.random.randint(self.n_arms)
        # Else, Select action with max Q(a) (Exploit) 
        else:
            # Breaking ties randomly
            return np.random.choice(np.flatnonzero(self.q_values == self.q_values.max()))

    def update(self, action, reward):
        # N(a) <- N(a) + 1 
        self.action_counts[action] += 1
        
        # Update estimate: Q(a) <- Q(a) + 1/N(a) * (R - Q(a)) 
        step_size = 1.0 / self.action_counts[action]
        self.q_values[action] += step_size * (reward - self.q_values[action])

    def step(self):
        action = self.select_action()
        # Simulate reward (Normal distribution around true mean)
        reward = np.random.normal(self.true_means[action], 1)
        self.update(action, reward)
        return reward

# Simulation
bandit = EpsilonGreedyBandit(n_arms=10, epsilon=0.1)
rewards = [bandit.step() for _ in range(1000)]
print(f"Estimated Q-values: {bandit.q_values}")