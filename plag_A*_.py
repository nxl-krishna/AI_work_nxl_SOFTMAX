import heapq
import re
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict, Optional

class PlagiarismDetector:
    """
    Analyzes two texts for plagiarism by aligning sentences using the A* search
    algorithm and calculating their similarity with Levenshtein distance.
    """

    def __init__(self, text_a: str, text_b: str, similarity_threshold: int = 1):
        """
        Initializes the detector with the two texts and a similarity threshold.

        Args:
            text_a: The first text document as a string.
            text_b: The second text document as a string.
            similarity_threshold: The maximum Levenshtein distance to be
                                  considered a potential plagiarism case.
        """
        self.sentences_a = self._process_text(text_a)
        self.sentences_b = self._process_text(text_b)
        self.threshold = similarity_threshold

    def _process_text(self, document: str) -> List[str]:
        """Tokenizes text into sentences and normalizes them for comparison."""
        sentences = sent_tokenize(document)
        # Normalize by converting to lowercase and removing non-alphanumeric/space characters.
        return [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]

    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """Calculates Levenshtein distance between two strings using dynamic programming."""
        m, n = len(str1), len(str2)
        dp_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp_matrix[i][0] = i
        for j in range(n + 1):
            dp_matrix[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp_matrix[i][j] = min(dp_matrix[i - 1][j] + 1,        # Cost of deletion
                                      dp_matrix[i][j - 1] + 1,        # Cost of insertion
                                      dp_matrix[i - 1][j - 1] + cost) # Cost of substitution
        return dp_matrix[m][n]

    def _heuristic_cost_estimate(self, idx_a: int, idx_b: int) -> int:
        """An admissible heuristic for A*: the difference in remaining sentences."""
        remaining_a = len(self.sentences_a) - idx_a
        remaining_b = len(self.sentences_b) - idx_b
        return abs(remaining_a - remaining_b)

    def _reconstruct_alignment(self, history: Dict, current: Tuple) -> List:
        """Backtracks from the goal node to reconstruct the optimal alignment path."""
        path = []
        while current in history:
            previous_node, move_details = history[current]
            path.append(move_details)
            current = previous_node
        return path[::-1] # Reverse to get the correct order

    def _align_documents(self) -> Optional[List[Tuple]]:
        """Finds the optimal sentence alignment using the A* search algorithm."""
        len_a, len_b = len(self.sentences_a), len(self.sentences_b)
        start_node = (0, 0)
        
        # Priority queue stores tuples of (f_score, (index_a, index_b))
        priority_queue = [(0, start_node)]
        
        # history tracks where each node came from: {child: (parent, move_info)}
        history: Dict[Tuple, Tuple] = {}

        # g_scores track the actual cost from the start to the current node
        g_scores = {(i, j): float('inf') for i in range(len_a + 1) for j in range(len_b + 1)}
        g_scores[start_node] = 0

        while priority_queue:
            _, current_node = heapq.heappop(priority_queue)
            i, j = current_node

            if i == len_a and j == len_b: # Goal reached
                return self._reconstruct_alignment(history, current_node)

            # Explore neighbors: align, skip sentence in A, or skip sentence in B
            possible_moves = []
            if i < len_a and j < len_b: possible_moves.append("align")
            if i < len_a: possible_moves.append("skip_a")
            if j < len_b: possible_moves.append("skip_b")

            for move in possible_moves:
                cost, neighbor, move_info = 0, None, None
                
                if move == "align":
                    cost = self._calculate_edit_distance(self.sentences_a[i], self.sentences_b[j])
                    neighbor = (i + 1, j + 1)
                    move_info = (i, j, cost)
                elif move == "skip_a":
                    cost = 10  # A higher penalty for skipping a sentence
                    neighbor = (i + 1, j)
                    move_info = (i, -1, cost)
                elif move == "skip_b":
                    cost = 10
                    neighbor = (i, j + 1)
                    move_info = (-1, j, cost)

                tentative_g = g_scores[current_node] + cost
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    history[neighbor] = (current_node, move_info)
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic_cost_estimate(neighbor[0], neighbor[1])
                    heapq.heappush(priority_queue, (f_score, neighbor))
        
        return None # Should not be reached if a path exists

    def find_suspicious_pairs(self) -> List[Dict]:
        """
        Runs the full analysis and returns sentence pairs that are suspiciously similar.
        
        Returns:
            A list of dictionaries, each detailing a potentially plagiarized sentence pair.
        """
        alignment = self._align_documents()
        if not alignment:
            return []

        plagiarized_pairs = []
        for idx_a, idx_b, cost in alignment:
            # A direct alignment (not a skip) with a cost below the threshold is suspicious.
            if idx_a != -1 and idx_b != -1 and cost <= self.threshold:
                pair_details = {
                    "original_sentence": self.sentences_a[idx_a],
                    "suspicious_sentence": self.sentences_b[idx_b],
                    "edit_distance": cost
                }
                plagiarized_pairs.append(pair_details)
        
        return plagiarized_pairs

if __name__ == "__main__":
    doc_original = "Artificial intelligence is a branch of computer science. It involves the creation of intelligent machines."
    doc_suspicious = "AI is a field in computer science. It is about making smart machines that work like humans."

    print(f"Original Document:\n'{doc_original}'\n")
    print(f"Suspicious Document:\n'{doc_suspicious}'\n")
    
    # Instantiate the detector with a higher threshold to catch modifications.
    detector = PlagiarismDetector(doc_original, doc_suspicious, similarity_threshold=20)
    
    # Find and display the results.
    results = detector.find_suspicious_pairs()

    print("--- Plagiarism Analysis Results ---")
    if not results:
        print("No sentences were found to be suspiciously similar based on the threshold.")
    else:
        for i, result in enumerate(results):
            print(f"Suspicious Pair #{i + 1}:")
            print(f"  Original:   '{result['original_sentence']}'")
            print(f"  Suspicious: '{result['suspicious_sentence']}'")
            print(f"  --> Edit Distance: {result['edit_distance']} (Threshold: {detector.threshold})\n")
