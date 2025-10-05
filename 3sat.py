
import random
import itertools

# --- Heuristic and Helper Functions ---

def count_satisfied_clauses(formula, assignment):
    """Heuristic function: counts clauses satisfied by the assignment."""
    count = 0
    for clause in formula:
        for literal in clause:
            var = literal.lower()
            # True if: positive literal and assigned True OR negative literal and assigned False
            if (literal.islower() and assignment[var]) or \
               (literal.isupper() and not assignment[var]):
                count += 1
                break
    return count

# --- Neighborhood Functions for VND ---

def get_neighborhood_flip1(assignment):
    """N1: Generates all neighbors by flipping one variable."""
    neighbors = []
    for var in assignment:
        neighbor = assignment.copy()
        neighbor[var] = not neighbor[var]
        neighbors.append(neighbor)
    return neighbors

def get_neighborhood_focused_flip(formula, assignment):
    """N2: Flips a variable in a random unsatisfied clause."""
    unsatisfied = []
    for clause in formula:
        is_satisfied = False
        for literal in clause:
            var = literal.lower()
            if (literal.islower() and assignment[var]) or \
               (literal.isupper() and not assignment[var]):
                is_satisfied = True
                break
        if not is_satisfied:
            unsatisfied.append(clause)

    if not unsatisfied:
        return []

    clause_to_flip = random.choice(unsatisfied)
    var_to_flip = random.choice(clause_to_flip).lower()
    
    neighbor = assignment.copy()
    neighbor[var_to_flip] = not neighbor[var_to_flip]
    return [neighbor]

def get_neighborhood_flip2(assignment, max_neighbors=50):
    """N3: Generates neighbors by flipping two variables."""
    neighbors = []
    vars_to_flip = list(itertools.combinations(assignment.keys(), 2))
    # To avoid a huge neighborhood, sample from the possible pairs
    if len(vars_to_flip) > max_neighbors:
        vars_to_flip = random.sample(vars_to_flip, max_neighbors)

    for var1, var2 in vars_to_flip:
        neighbor = assignment.copy()
        neighbor[var1] = not neighbor[var1]
        neighbor[var2] = not neighbor[var2]
        neighbors.append(neighbor)
    return neighbors

# --- Local Search Algorithms ---

def hill_climbing(formula, variables, max_restarts=10):
    """Hill-Climbing algorithm with random restarts."""
    best_assignment = None
    max_score = -1

    for _ in range(max_restarts):
        current_assignment = {v: random.choice([True, False]) for v in variables}
        while True:
            current_score = count_satisfied_clauses(formula, current_assignment)
            if current_score > max_score:
                max_score = current_score
                best_assignment = current_assignment
            
            if current_score == len(formula):
                return best_assignment, max_score

            neighbors = get_neighborhood_flip1(current_assignment)
            best_neighbor, best_neighbor_score = None, current_score
            for n in neighbors:
                score = count_satisfied_clauses(formula, n)
                if score > best_neighbor_score:
                    best_neighbor, best_neighbor_score = n, score
            
            if best_neighbor is None:
                break
            current_assignment = best_neighbor
            
    return best_assignment, max_score

def beam_search(formula, variables, beam_width=3):
    """Beam Search algorithm."""
    beam = [{v: random.choice([True, False]) for v in variables} for _ in range(beam_width)]

    for _ in range(len(variables) * 2): # Limit search depth
        all_successors = []
        for assignment in beam:
            score = count_satisfied_clauses(formula, assignment)
            if score == len(formula):
                return assignment, score
            all_successors.extend(get_neighborhood_flip1(assignment))

        if not all_successors:
            break
            
        all_successors.sort(key=lambda a: count_satisfied_clauses(formula, a), reverse=True)
        beam = all_successors[:beam_width]

    best_in_beam = max(beam, key=lambda a: count_satisfied_clauses(formula, a))
    return best_in_beam, count_satisfied_clauses(formula, best_in_beam)

def variable_neighborhood_descent(formula, variables):
    """Variable Neighborhood Descent (VND) algorithm."""
    current_assignment = {v: random.choice([True, False]) for v in variables}
    neighborhood_functions = [
        get_neighborhood_flip1,
        get_neighborhood_focused_flip,
        get_neighborhood_flip2
    ]
    
    k = 0
    while k < len(neighborhood_functions):
        current_score = count_satisfied_clauses(formula, current_assignment)
        if current_score == len(formula):
            return current_assignment, current_score

        # Select neighborhood based on k
        if k == 1: # Focused flip needs the formula
            neighborhood = neighborhood_functions[k](formula, current_assignment)
        else:
            neighborhood = neighborhood_functions[k](current_assignment)
        
        if not neighborhood: # Focused flip might return empty
            k += 1
            continue

        best_neighbor = max(neighborhood, key=lambda n: count_satisfied_clauses(formula, n))
        best_neighbor_score = count_satisfied_clauses(formula, best_neighbor)

        if best_neighbor_score > current_score:
            current_assignment = best_neighbor
            k = 0  # Go back to the first neighborhood
        else:
            k += 1 # Try the next neighborhood structure
            
    return current_assignment, count_satisfied_clauses(formula, current_assignment)

# --- Main Execution Block ---
if __name__ == '__main__':
    # Generate a sample 3-SAT problem
    # A hard instance for 20 variables is often around 85 clauses (ratio ~4.25)
    N_VARS, M_CLAUSES, K_LITERALS = 20, 85, 3
    variables_list = list(ascii_lowercase[:N_VARS])
    
    print(f"Generating a {K_LITERALS}-SAT problem with {N_VARS} vars and {M_CLAUSES} clauses...")
    formula_instance = []
    for _ in range(M_CLAUSES):
        chosen = random.sample(variables_list, K_LITERALS)
        clause = [random.choice([v, v.upper()]) for v in chosen]
        formula_instance.append(clause)
    print("Problem generated.\n")

    # --- Run Hill-Climbing ---
    print("--- Running Hill-Climbing ---")
    hc_assign, hc_score = hill_climbing(formula_instance, variables_list)
    print(f"Result: {hc_score}/{M_CLAUSES} clauses satisfied.")
    # print(f"Assignment: {hc_assign}")
    print("-" * 30 + "\n")

    # --- Run Beam Search (Beam Width 3) ---
    print("--- Running Beam Search (Width=3) ---")
    bs3_assign, bs3_score = beam_search(formula_instance, variables_list, beam_width=3)
    print(f"Result: {bs3_score}/{M_CLAUSES} clauses satisfied.")
    # print(f"Assignment: {bs3_assign}")
    print("-" * 30 + "\n")

    # --- Run Beam Search (Beam Width 4) ---
    print("--- Running Beam Search (Width=4) ---")
    bs4_assign, bs4_score = beam_search(formula_instance, variables_list, beam_width=4)
    print(f"Result: {bs4_score}/{M_CLAUSES} clauses satisfied.")
    # print(f"Assignment: {bs4_assign}")
    print("-" * 30 + "\n")

    # --- Run Variable Neighborhood Descent ---
    print("--- Running Variable Neighborhood Descent ---")
    vnd_assign, vnd_score = variable_neighborhood_descent(formula_instance, variables_list)
    print(f"Result: {vnd_score}/{M_CLAUSES} clauses satisfied.")
    # print(f"Assignment: {vnd_assign}")
    print("-" * 30 + "\n")
