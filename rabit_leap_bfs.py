from collections import deque

class PuzzleSolver:
    """
    A class to solve the 'rabbits' puzzle using Breadth-First Search.
    
    This puzzle involves moving entities ('e' and 'w') across a line
    to reverse their initial positions.
    """
    def __init__(self, initial_config, final_config):
        """
        Initializes the solver with the start and goal configurations.
        
        Args:
            initial_config (str): The starting layout of the puzzle.
            final_config (str): The target layout of the puzzle.
        """
        # Using tuples for immutable and hashable state representation
        self.start_node = tuple(initial_config)
        self.goal_node = tuple(final_config)
        self.state_length = len(self.start_node)

    def _generate_successors(self, current_node):
        """
        Generates all valid and reachable next states from the current state.

        A move is valid if an entity 'e' moves right or an entity 'w' moves left,
        either by sliding into an adjacent empty space or jumping over one entity.
        
        Args:
            current_node (tuple): The current state tuple (e.g., ('e', ' ', 'w')).

        Yields:
            tuple: A valid successor state.
        """
        empty_idx = current_node.index(' ')
        
        # Potential moves are defined by the entity and the distance to move.
        # 'e' moves right (+1, +2), 'w' moves left (-1, -2).
        # We check positions relative to the empty space.
        possible_moves = {
            # Check for 'w' to the right of the space
            empty_idx + 1: 'w',
            empty_idx + 2: 'w',
            # Check for 'e' to the left of the space
            empty_idx - 1: 'e',
            empty_idx - 2: 'e'
        }

        for from_idx, entity_type in possible_moves.items():
            # Ensure the index is valid and the correct entity is present
            if 0 <= from_idx < self.state_length and current_node[from_idx] == entity_type:
                new_node_list = list(current_node)
                # Swap the entity with the empty space
                new_node_list[empty_idx], new_node_list[from_idx] = new_node_list[from_idx], new_node_list[empty_idx]
                yield tuple(new_node_list)

    def _reconstruct_path(self, history, final_node):
        """
        Builds the solution path by backtracking from the goal node.
        
        Args:
            history (dict): A dictionary mapping each state to its predecessor.
            final_node (tuple): The goal state.

        Returns:
            list: A list of strings representing the path from start to goal.
        """
        path = []
        current = final_node
        while current is not None:
            path.append("".join(current))
            current = history.get(current) # Backtrack to the parent
        return path[::-1] # Reverse for chronological order

    def find_solution(self):
        """
        Executes the Breadth-First Search algorithm to find the shortest solution path.

        Returns:
            list: The sequence of states from start to goal, or None if no solution exists.
        """
        # The frontier stores nodes to be explored
        frontier = deque([self.start_node])
        # Explored stores nodes already visited to prevent cycles
        explored = {self.start_node}
        # History tracks the path: {child_node: parent_node}
        history = {self.start_node: None}

        while frontier:
            current_node = frontier.popleft()

            if current_node == self.goal_node:
                return self._reconstruct_path(history, current_node)

            for successor in self._generate_successors(current_node):
                if successor not in explored:
                    explored.add(successor)
                    history[successor] = current_node
                    frontier.append(successor)
        
        return None # Return None if the frontier is exhausted

def main():
    """
    Main function to set up and run the puzzle solver.
    """
    start_configuration = "eee www"
    goal_configuration = "www eee"

    solver = PuzzleSolver(start_configuration, goal_configuration)
    solution = solver.find_solution()

    if solution:
        print("✅ Solution Found! Here are the steps:")
        for i, step in enumerate(solution):
            print(f"  Step {i:02d}: [{step}]")
    else:
        print("❌ A solution was not found.")

if __name__ == "__main__":
    main()
