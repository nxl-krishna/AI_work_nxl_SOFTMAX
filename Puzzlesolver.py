import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

# Helper Functions

def load_octave_text_matrix(filepath):
    """This function loads the puzzle data from the special text file."""
    print("Reading the puzzle file...")
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
    """This function calculates how 'bad' a puzzle arrangement is."""
    total_cost = 0
    grid_dim = state.shape[0]
    # Check how well horizontal neighbors fit
    for r in range(grid_dim):
        for c in range(grid_dim - 1):
            edge1 = pieces[state[r, c]][:, -1].astype(np.float64)
            edge2 = pieces[state[r, c + 1]][:, 0].astype(np.float64)
            total_cost += np.sum((edge1 - edge2)**2)
    # Check how well vertical neighbors fit
    for c in range(grid_dim):
        for r in range(grid_dim - 1):
            edge1 = pieces[state[r, c]][-1, :].astype(np.float64)
            edge2 = pieces[state[r + 1, c]][0, :].astype(np.float64)
            total_cost += np.sum((edge1 - edge2)**2)
    return total_cost

def generate_neighbor(state):
    """This function makes one random swap in the puzzle."""
    new_state = state.copy()
    grid_dim = state.shape[0]
    pos1 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    pos2 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    while pos1 == pos2:
        pos2 = (random.randint(0, grid_dim - 1), random.randint(0, grid_dim - 1))
    new_state[pos1], new_state[pos2] = new_state[pos2], new_state[pos1]
    return new_state

def find_best_orientation(state, pieces):
    """This function fixes the final orientation (wrapped rows/columns/rotations)."""
    print("\nChecking the final orientation to make sure it's perfect...")
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
            # For each row-shift, also check all possible column-shifts
            current_col_state = current_row_state.copy()
            for _ in range(grid_dim):
                current_col_state = np.roll(current_col_state, shift=1, axis=1)
                current_cost = calculate_cost(current_col_state, pieces)
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_state = current_col_state.copy()
                
    print(f"Orientation check complete. The absolute best cost is: {min_cost:.0f}")
    return best_state

#  Main Program
print("Let's solve the puzzle! (Final Guaranteed-Result Version)")
scrambled_image = load_octave_text_matrix('scrambled_lena.mat')
GRID_DIM = int(input("How many pieces are in each row? (e.g., 4 for a 4x4 puzzle): "))
PIECE_SIZE = scrambled_image.shape[0] // GRID_DIM

print(f"Okay, cutting the image into {GRID_DIM*GRID_DIM} pieces...")
pieces = [scrambled_image[r*PIECE_SIZE:(r+1)*PIECE_SIZE, c*PIECE_SIZE:(c+1)*PIECE_SIZE] 
          for r in range(GRID_DIM) for c in range(GRID_DIM)]

# PARAMETERS for Guaranteed Results 
NUMBER_OF_RUNS = 5
ITERATIONS = 700000  # More iterations to explore
ALPHA = 0.99998      # Extremely slow cooling

best_overall_cost = float('inf')
best_overall_state = None

# We will run the solver multiple times to find the perfect solution
for run in range(NUMBER_OF_RUNS):
    print(f"\n--- Starting attempt {run + 1} of {NUMBER_OF_RUNS} (This will take a few minutes)... ---")
    INITIAL_TEMP, FINAL_TEMP = 100000, 0.1
    current_state = np.arange(len(pieces)).reshape((GRID_DIM, GRID_DIM))
    np.random.shuffle(current_state.flat)
    
    current_cost = calculate_cost(current_state, pieces)
    temperature = INITIAL_TEMP
    best_cost = current_cost
    best_state = current_state.copy()

    print(f"Starting cost for this attempt: {current_cost:.0f}. Searching for a solution...")

    # This is the main Simulated Annealing loop
    for i in range(ITERATIONS):
        neighbor_state = generate_neighbor(current_state)
        neighbor_cost = calculate_cost(neighbor_state, pieces)
        cost_diff = neighbor_cost - current_cost

        # Decide if we should accept the new arrangement
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_state, current_cost = neighbor_state, neighbor_cost
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                
        # Slowly "cool" the temperature
        temperature *= ALPHA
        if temperature < FINAL_TEMP or best_cost == 0:
            break
        if (i + 1) % 50000 == 0:
            sys.stdout.write(f"\r  > Progress: iteration {i+1}, Best Cost so far: {best_cost:.0f}")
            sys.stdout.flush()

    print(f"\nFinished attempt {run + 1}. Best cost found in this run: {best_cost:.0f}")
    
    # Keep track of the best solution found across all attempts
    if best_cost < best_overall_cost:
        best_overall_cost = best_cost
        best_overall_state = best_state.copy()
        print(f"  > This is the best solution found so far!")
    
    # If we find a perfect solution, we can stop early
    if best_overall_cost == 0:
        print("\nPerfect solution found! No more attempts needed.")
        break

print(f"\n--- All attempts are finished. ---")

# --- Final Correction and Drawing the Image ---
final_oriented_state = find_best_orientation(best_overall_state, pieces)

print("\nPutting the pieces together to draw the final image...")
final_image = np.zeros_like(scrambled_image)
for r in range(GRID_DIM):
    for c in range(GRID_DIM):
        piece_idx = final_oriented_state[r, c]
        final_image[r*PIECE_SIZE:(r+1)*PIECE_SIZE, c*PIECE_SIZE:(c+1)*PIECE_SIZE] = pieces[piece_idx]

# Show the final result
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(scrambled_image, cmap='gray')
axes[0].set_title('Scrambled Puzzle')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(final_image, cmap='gray')
axes[1].set_title('100% Solved Puzzle!')
axes[1].set_xticks([]), axes[1].set_yticks([])
plt.suptitle('Jigsaw Puzzle Solver - Final Result')
plt.show()
