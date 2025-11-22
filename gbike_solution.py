import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

MAX_BIKES = 20
MAX_MOVE = 5
GAMMA = 0.9
RENTAL_REWARD = 10
MOVE_COST = 2

# Poisson averages (Lambda)
# Requests: Loc1=3, Loc2=4
# Returns:  Loc1=3, Loc2=2
LAMBDA_REQ = [3, 4]
LAMBDA_RET = [3, 2]

# Precompute Poisson probabilities to speed up the code
# We truncate at 20 because probability of >20 is negligible for these lambdas
POISSON_UPPER_BOUND = 21
poisson_cache = {}

def get_poisson_prob(n, lam):
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

class GbikeRental:
    def __init__(self):
        # V is the Value function: 21x21 grid
        self.V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
        # Policy is the Action matrix: 21x21 grid
        self.policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

    def calculate_cost(self, action):
        """
        Part 3 Modification: The Employee Bus Rule
        - Moving from Loc 1 to Loc 2 (action > 0): First bike is free.
        - Moving from Loc 2 to Loc 1 (action < 0): Standard cost.
        """
        cost = 0
        if action > 0:  # Moving 1 to 2
            # First bike is free, pay for the rest
            # If action is 1, cost is 0. If 2, cost is 2.
            cost = MOVE_COST * (action - 1)
        else:  # Moving 2 to 1 (action is negative or 0)
            cost = MOVE_COST * abs(action)
        return cost

    def expected_return(self, s1, s2, action, V_current):
        """
        Calculates the expected return for a specific state and action.
        """
        # 1. Move bikes (Action)
        # Check boundaries: Can't move more than you have
        real_action = action
        if action > 0: # Moving 1 to 2
            real_action = min(action, s1) 
        else: # Moving 2 to 1
            real_action = max(action, -s2)
        
        # State after moving (Overnight state)
        s1_overnight = s1 - real_action
        s2_overnight = s2 + real_action
        
        # Clip to max capacity (20) just in case, though logic should prevent it
        s1_overnight = min(s1_overnight, MAX_BIKES)
        s2_overnight = min(s2_overnight, MAX_BIKES)

        # If more than 10 bikes kept overnight, pay 4 INR penalty
        parking_penalty = 0
        if s1_overnight > 10: parking_penalty += 4
        if s2_overnight > 10: parking_penalty += 4

        # Initial Cost (Move cost + Parking penalty)
        expected_reward = -self.calculate_cost(real_action) - parking_penalty

        # 2. Calculate expected reward from rentals (Stochastic)
        # We iterate through possible Request/Return scenarios
        # To save time, we iterate over a reasonable range where prob > epsilon
        
        # This looks computationally expensive (4 nested loops), but we can simplify.
        # The transitions for Loc 1 and Loc 2 are independent given the overnight state.
        
        reward_1 = 0
        transition_probs_1 = np.zeros(MAX_BIKES + 1)
        for req in range(13): # Truncate Poisson at reasonable limit
            prob_req = get_poisson_prob(req, LAMBDA_REQ[0])
            
            valid_rentals = min(s1_overnight, req)
            reward_1 += prob_req * (valid_rentals * RENTAL_REWARD)
            
            bikes_after_rent = s1_overnight - valid_rentals
            
            for ret in range(13):
                prob_ret = get_poisson_prob(ret, LAMBDA_RET[0])
                prob = prob_req * prob_ret
                
                final_bikes = min(bikes_after_rent + ret, MAX_BIKES)
                transition_probs_1[final_bikes] += prob

        reward_2 = 0
        transition_probs_2 = np.zeros(MAX_BIKES + 1)
        for req in range(13):
            prob_req = get_poisson_prob(req, LAMBDA_REQ[1])
            
            valid_rentals = min(s2_overnight, req)
            reward_2 += prob_req * (valid_rentals * RENTAL_REWARD)
            
            bikes_after_rent = s2_overnight - valid_rentals
            
            for ret in range(13):
                prob_ret = get_poisson_prob(ret, LAMBDA_RET[1])
                prob = prob_req * prob_ret
                
                final_bikes = min(bikes_after_rent + ret, MAX_BIKES)
                transition_probs_2[final_bikes] += prob
        expected_reward += reward_1 + reward_2

        # 3. Add Discounted Future Value
        # We use the transition probabilities we just summed up
        # V(s) = R + gamma * sum(P(s'|s,a) * V(s'))
        
        # We can do this efficiently with outer product
        # Joint probability P(s1', s2') = P(s1') * P(s2')
        joint_probs = np.outer(transition_probs_1, transition_probs_2)
        future_value = np.sum(joint_probs * V_current)
        
        return expected_reward + GAMMA * future_value

    def policy_evaluation(self):
        print("Evaluating Policy...")
        while True:
            delta = 0
            V_old = self.V.copy()
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    action = self.policy[i, j]
                    self.V[i, j] = self.expected_return(i, j, action, V_old)
                    delta = max(delta, abs(V_old[i, j] - self.V[i, j]))
            
            print(f"  Delta: {delta:.4f}")
            if delta < 1e-2:
                break

    def policy_improvement(self):
        print("Improving Policy...")
        policy_stable = True
        
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                old_action = self.policy[i, j]
                
                action_returns = []
                for action in self.actions:
                    if (action > 0 and action > i) or (action < 0 and abs(action) > j):
                        action_returns.append(-float('inf'))
                    else:
                        action_returns.append(self.expected_return(i, j, action, self.V))
                
                best_action = self.actions[np.argmax(action_returns)]
                self.policy[i, j] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable

    def run(self):
        iterations = 0
        while True:
            print(f"--- Iteration {iterations} ---")
            self.policy_evaluation()
            stable = self.policy_improvement()
            iterations += 1
            if stable:
                print("Policy Converged!")
                break
        return self.policy, self.V

if __name__ == "__main__":
    agent = GbikeRental()
    final_policy, final_value = agent.run()

    plt.figure(figsize=(10, 8))
    sns.heatmap(final_policy, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Optimal Policy (Modified: Employee + Parking)")
    plt.xlabel("Bikes at Location 2")
    plt.ylabel("Bikes at Location 1")
    plt.gca().invert_yaxis() 
    plt.show()
    
    np.savetxt("optimal_policy.csv", final_policy, delimiter=",")
    print("Policy saved to optimal_policy.csv")
