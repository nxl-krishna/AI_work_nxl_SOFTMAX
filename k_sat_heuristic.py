import random
from string import ascii_lowercase
from typing import List

class KSATProblemGenerator:
    """
    Generates a random k-SAT problem instance using an object-oriented approach.
    """

    def __init__(self, k: int, m: int, n: int):
        """
        Initializes the generator with k-SAT problem parameters.

        Args:
            k: The number of literals per clause.
            m: The number of clauses in the formula.
            n: The number of unique variables.
        """
        if k > n:
            raise ValueError(f"Literals per clause (k={k}) cannot exceed variables (n={n}).")
        if n > 26:
            raise ValueError("Generator supports a maximum of 26 variables (a-z).")

        self.k = k
        self.num_clauses = m
        self.num_vars = n
        self.variables = list(ascii_lowercase[:n])
        self.problem: List[List[str]] = []

    def generate(self) -> List[List[str]]:
        """
        Constructs the k-SAT problem.

        The algorithm first selects k unique variables, then randomly assigns their
        polarity (positive/negative) to form a clause.

        Returns:
            A list of lists, where each inner list represents a clause.
        """
        self.problem = []
        for _ in range(self.num_clauses):
            chosen_vars = random.sample(self.variables, self.k)
            clause = [random.choice([var, var.upper()]) for var in chosen_vars]
            self.problem.append(clause)
        return self.problem

    def __str__(self) -> str:
        """Provides a user-friendly string representation of the problem."""
        if not self.problem:
            return "No k-SAT problem generated. Call .generate() first."

        header = f"--- Generated {self.k}-SAT Problem ({self.num_clauses} clauses, {self.num_vars} vars) ---\n"
        clauses_str = "\n".join(
            f"  Clause {i+1}: ( {' v '.join(clause)} )"
            for i, clause in enumerate(self.problem)
        )
        return header

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🤖 k-SAT Problem Generator 🤖")
    try:
        k_val = int(input("Enter literals per clause (k): "))
        m_val = int(input("Enter number of clauses (m): "))
        n_val = int(input("Enter total number of variables (n): "))

        generator = KSATProblemGenerator(k=k_val, m=m_val, n=n_val)
        problem_instance = generator.generate()
        print("\n" + str(generator))

    except ValueError as e:
        print(f"\n❌ Input Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
