class DFSPuzzleSolver:
    """
    Solves the 'rabbits' puzzle using an iterative Depth-First Search (DFS).
    
    This implementation uses an explicit stack to traverse the state space,
    avoiding recursion and the use of global variables.
    """
    def __init__(self, initial_config, final_config):
        """
        Initializes the solver with start and goal configurations.

        Args:
            initial_config (str): The starting layout of the puzzle.
            final_config (str): The target layout of the puzzle.
        """
        self.start_node = tuple(initial_config)
        self.goal_node = tuple(final_config)
        self.state_length = len(self.start_node)
        self.nodes_visited = 0

    def _get_successors(self, node):
        """
        Generates all valid successor states from a given node.
        
        A 'w' can only move left, and an 'e' can only move right.
        Moves can be a single step into an empty space or a jump over one character.

        Args:
            node (tuple): The current state tuple.

        Returns:
            list: A list of valid successor state tuples.
        """
        successors = []
        blank_pos = node.index(' ')

        # Check for a leftward move (a 'w' moving into the blank)
        # This can be a slide from one position away or a jump from two positions away.
        for offset in [1, 2]:
            from_pos = blank_pos + offset
            if from_pos < self.state_length and node[from_pos] == 'w':
                new_node_list = list(node)
                new_node_list[blank_pos], new_node_list[from_pos] = new_node_list[from_pos], new_node_list[blank_pos]
                successors.append(tuple(new_node_list))

        # Check for a rightward move (an 'e' moving into the blank)
        for offset in [1, 2]:
            from_pos = blank_pos - offset
            if from_pos >= 0 and node[from_pos] == 'e':
                new_node_list = list(node)
                new_node_list[blank_pos], new_node_list[from_pos] = new_node_list[from_pos], new_node_list[blank_pos]
                successors.append(tuple(new_node_list))
        
        return successors

    def find_path(self):
        """
        Executes the iterative DFS algorithm to find a solution path.

        Returns:
            list: A sequence of strings representing the path, or None if no solution is found.
        """
        # The stack will hold tuples of (node, path_to_that_node)
        stack = [(self.start_node, [self.start_node])]
        explored_nodes = {self.start_node}

        while stack:
            current_node, current_path = stack.pop() # LIFO behavior for DFS

            if current_node == self.goal_node:
                self.nodes_visited = len(explored_nodes)
                # Convert the path of tuples back to a path of strings
                return ["".join(node) for node in current_path]

            # Explore neighbors
            for successor in self._get_successors(current_node):
                if successor not in explored_nodes:
                    explored_nodes.add(successor)
                    new_path = current_path + [successor]
                    stack.append((successor, new_path))
        
        self.nodes_visited = len(explored_nodes)
        return None # No solution found

def main():
    """
    Main execution function to set up and run the puzzle solver.
    """
    # Note: Corrected typo from "intialState" to "initial_state"
    initial_state = "wee wwe"
    final_state = "www eee"

    solver = DFSPuzzleSolver(initial_state, final_state)
    solution_path = solver.find_path()

    if solution_path:
        print("✅ Path Found!")
        print(solution_path)
    else:
        print("❌ A solution could not be found.")

    print(f"Number of states discovered: {solver.nodes_visited}")

# Standard entry point
if __name__ == "__main__":
    main()
