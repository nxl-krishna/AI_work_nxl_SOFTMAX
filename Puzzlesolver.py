import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

# --- Helper Functions (The Simple, Correct Versions) ---

def load_octave_text_matrix(filepath):
    """Loads the puzzle matrix from the specific Octave text format."""
    print("Reading file with our custom loader...")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_start_line = 0
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            data_start_line = i
            break
    dims = lines[data_start_line].split()
    rows, cols = int(dims[0]), int(dims[1])
    pixel_data_lines = lines[data_start_line + 1:]
    image_array = np.fromstring(''.join(pixel_data_lines), dtype=np.uint8, sep=' ')
    image = image_array.reshape((rows, cols))
    return image

def calculate_cost(state, pieces):
    """Calculates the cost based on piece positions only (no rotation)."""
    total_cost = 0
    grid_dim = state.shape[0]
    # Horizontal cost
    for r in range(grid_dim):
        for c in range(grid_dim - 1):
            edge1 = pieces[state[r, c]][:, -1].astype(np.float64)
            edge2 = pieces[state[r, c + 1]][:, 0].astype(np.float64)
            total_cost += np.sum((edge1 - edge2)**2)
    # Vertical cost
    for c in range(grid_dim):
        for r in range(grid_dim - 1):
            edge1 = pieces[state[r, c]][-1, :].astype(np.float64)
            edge2 = pieces[state[r + 1, c]][0, :].astype(np.float64)
            total_cost += np.sum((edge1 - edge2)**2)
    return total_cost

def generate_neighbor(state):
    """Generates a neighbor by swapping two random pieces."""
    new_state = state.copy()
    grid_dim = state.shape[0]
    pos1 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    pos2 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    while pos1 == pos2:
        pos2 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    new_state[pos1], new_state[pos2] = new_state[pos2], new_state[pos1]
    return new_state

def find_best_orientation(state, pieces):
    """
    Post-processing step to fix all 'wrapped' or 'rotated' solutions.
    """
    print("\nChecking for wrapped/rotated solutions...")
    best_state = state.copy()
    min_cost = calculate_cost(state, pieces)
    grid_dim = state.shape[0]
    
    # Check all 4 possible 90-degree rotations
    for k in range(4):
        rotated_state = np.rot90(state, k)
        
        # For each rotation, check all possible row-shifts
        current_row_state = rotated_state.copy()
        for _ in range(grid_dim):
            current_row_state = np.roll(current_row_state, shift=1, axis=0)
            
            # --- THE FIX IS HERE ---
            # For each row-shift, also check all possible column-shifts
            current_col_state = current_row_state.copy()
            for _ in range(grid_dim):
                current_col_state = np.roll(current_col_state, shift=1, axis=1) # axis=1 is for columns
                current_cost = calculate_cost(current_col_state, pieces)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_state = current_col_state.copy()
                
    print(f"Orientation check complete. Final best cost: {min_cost:.0f}")
    return best_state

# --- Main Program ---
print("Let's solve the puzzle! (Final Simplified Version)")
scrambled_image = load_octave_text_matrix('scrambled_lena.mat')
GRID_DIM = int(input("How many pieces are in each row? (e.g., 4 for a 4x4 puzzle): "))
PIECE_SIZE = scrambled_image.shape[0] // GRID_DIM

print(f"Cutting image into {GRID_DIM*GRID_DIM} pieces...")
pieces = [scrambled_image[r*PIECE_SIZE:(r+1)*PIECE_SIZE, c*PIECE_SIZE:(c+1)*PIECE_SIZE] 
          for r in range(GRID_DIM) for c in range(GRID_DIM)]

# --- PATIENT PARAMETERS for the Simple Solver ---
NUMBER_OF_RUNS = 5
ITERATIONS = 500000
ALPHA = 0.99995

best_overall_cost = float('inf')
best_overall_state = None

for run in range(NUMBER_OF_RUNS):
    print(f"\n--- Starting Run {run + 1}/{NUMBER_OF_RUNS} ---")
    INITIAL_TEMP, FINAL_TEMP = 100000, 0.1
    current_state = np.arange(len(pieces)).reshape((GRID_DIM, GRID_DIM))
    np.random.shuffle(current_state.flat)
    
    current_cost = calculate_cost(current_state, pieces)
    temperature = INITIAL_TEMP
    best_cost = current_cost
    best_state = current_state.copy()

    print(f"Initial Cost for Run {run+1}: {current_cost:.0f}. Working...")

    for i in range(ITERATIONS):
        neighbor_state = generate_neighbor(current_state)
        neighbor_cost = calculate_cost(neighbor_state, pieces)
        cost_diff = neighbor_cost - current_cost

        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_state, current_cost = neighbor_state, neighbor_cost
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                
        temperature *= ALPHA
        if temperature < FINAL_TEMP or best_cost == 0:
            break
        if (i + 1) % 50000 == 0:
            sys.stdout.write(f"\r  > Iteration {i+1}, Best Cost: {best_cost:.0f}")
            sys.stdout.flush()

    print(f"\nFinished Run {run + 1}. Best cost: {best_cost:.0f}")
    if best_cost < best_overall_cost:
        best_overall_cost = best_cost
        best_overall_state = best_state.copy()
        print(f"*** New best overall solution found! Cost: {best_overall_cost:.0f} ***")
    
    if best_overall_cost == 0:
        print("Perfect solution found! Stopping early.")
        break

print(f"\n--- All Runs Finished. ---")

# --- Final Correction and Visualization ---
final_oriented_state = find_best_orientation(best_overall_state, pieces)

print("\nReconstructing final image...")
final_image = np.zeros_like(scrambled_image)
for r in range(GRID_DIM):
    for c in range(GRID_DIM):
        piece_idx = final_oriented_state[r, c]
        final_image[r*PIECE_SIZE:(r+1)*PIECE_SIZE, c*PIECE_SIZE:(c+1)*PIECE_SIZE] = pieces[piece_idx]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(scrambled_image, cmap='gray')
axes[0].set_title('Scrambled Puzzle')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(final_image, cmap='gray')
axes[1].set_title('100% Solved Puzzle!')
axes[1].set_xticks([]), axes[1].set_yticks([])
plt.suptitle('Jigsaw Puzzle Solver - Final Result')
plt.show()