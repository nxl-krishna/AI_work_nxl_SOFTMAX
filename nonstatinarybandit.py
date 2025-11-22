import numpy as np
import matplotlib.pyplot as plt

def run_non_stationary_experiment(n_arms=10, steps=10000, epsilon=0.1, alpha=0.1):
    """
    Simulates the Non-Stationary Bandit problem described in Sections III and IV.
    """
    
    # --- INITIALIZATION ---
    # 1. Initialize Q(a) = 0 for all actions (Agent's estimates) [cite: 44]
    q_estimates = np.zeros(n_arms)
    
    # 2. Initialize True Means equal at the start (e.g., all 0) [cite: 52]
    true_means = np.zeros(n_arms)
    
    # To track performance for plotting
    rewards_history = []
    optimal_action_counts = []

    # --- TIME STEP LOOP ---
    for t in range(steps):
        
        # --- STEP 1: Environment Changes (The "Non-Stationary" part) ---
        # "The mean reward for each arm changes according to a random walk" [cite: 57]
        # "Adding a normally distributed increment (mean 0, std 0.01)" 
        random_walk_noise = np.random.normal(loc=0.0, scale=0.01, size=n_arms)
        true_means += random_walk_noise
        
        # Identify which arm is actually the best right now (for analysis)
        best_arm_idx = np.argmax(true_means)

        # --- STEP 2: Action Selection (Epsilon-Greedy) ---
        # "Explore: Select a random action with probability e" [cite: 42]
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
        else:
            # "Exploit: Select the action with the highest estimated reward Q(a)" [cite: 42]
            # (Using random tie-breaking for robustness)
            max_q = np.max(q_estimates)
            candidates = np.where(q_estimates == max_q)[0]
            action = np.random.choice(candidates)

        # --- STEP 3: Reward Observation ---
        # Generate reward from normal distribution around the *current* true mean
        reward = np.random.normal(loc=true_means[action], scale=1.0)
        
        # --- STEP 4: Agent Update (Constant Step-Size Alpha) ---
        # "Q_{n+1} = Q_n + alpha * [R_n - Q_n]" 
        # This handles the non-stationarity better than the sample average.
        old_estimate = q_estimates[action]
        q_estimates[action] = old_estimate + alpha * (reward - old_estimate)

        # --- Data Logging ---
        rewards_history.append(reward)
        is_optimal = 1 if action == best_arm_idx else 0
        optimal_action_counts.append(is_optimal)

    return rewards_history, optimal_action_counts

# --- EXECUTION ---
# Running the experiment
print("Running Non-Stationary Bandit Experiment...")
rewards, optimal_picks = run_non_stationary_experiment(steps=2000, epsilon=0.1, alpha=0.1)

# Calculate average reward over the last 500 steps to see stability
avg_score = np.mean(rewards[-500:])
print(f"Experiment Complete.")
print(f"Average Reward (last 500 steps): {avg_score:.4f}")
print(f"Optimal Action % (last 500 steps): {np.mean(optimal_picks[-500:]) * 100:.2f}%")

# Optional: Visualization code (if running in a notebook)
# plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
# plt.xlabel('Steps')
# plt.ylabel('Average Reward')
# plt.show()