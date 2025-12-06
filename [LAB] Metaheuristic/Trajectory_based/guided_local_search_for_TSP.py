import numpy as np
import random
from typing import List, Tuple

class GuidedLocalSearch:
    def __init__(self, distances: np.ndarray, lambda_param: float = 0.3):
        """
        Initialize GLS for TSP.
        
        Args:
            distances: Distance matrix (n x n)
            lambda_param: Penalty parameter (typically 0.1-0.5)
        """
        self.distances = distances
        self.n = len(distances)
        self.lambda_param = lambda_param
        self.penalties = np.zeros((self.n, self.n))
        
    def calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate total cost of a tour including penalties."""
        cost = 0
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            cost += self.distances[city_a][city_b]
            cost += self.lambda_param * self.penalties[city_a][city_b]
        return cost
    
    def calculate_actual_cost(self, tour: List[int]) -> float:
        """Calculate actual tour cost without penalties."""
        cost = 0
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            cost += self.distances[city_a][city_b]
        return cost
    
    def two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Perform 2-opt swap."""
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
    def local_search(self, tour: List[int]) -> List[int]:
        """2-opt local search."""
        improved = True
        best_tour = tour.copy()
        best_cost = self.calculate_tour_cost(best_tour)
        
        while improved:
            improved = False
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_tour = self.two_opt_swap(best_tour, i, j)
                    new_cost = self.calculate_tour_cost(new_tour)
                    
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour
    
    def update_penalties(self, tour: List[int]):
        """Update penalties for edges in the tour."""
        # Find edges with maximum utility
        max_utility = -float('inf')
        edges_to_penalize = []
        
        for i in range(len(tour)):
            city_a = tour[i]
            city_b = tour[(i + 1) % len(tour)]
            
            # Utility = cost / (1 + penalty)
            utility = self.distances[city_a][city_b] / (1 + self.penalties[city_a][city_b])
            
            if utility > max_utility:
                max_utility = utility
                edges_to_penalize = [(city_a, city_b)]
            elif abs(utility - max_utility) < 1e-10:
                edges_to_penalize.append((city_a, city_b))
        
        # Increment penalties for selected edges
        for city_a, city_b in edges_to_penalize:
            self.penalties[city_a][city_b] += 1
            self.penalties[city_b][city_a] += 1
    
    def solve(self, max_iterations: int = 1000, initial_tour: List[int] = None) -> Tuple[List[int], float]:
        """
        Solve TSP using Guided Local Search.
        
        Args:
            max_iterations: Maximum number of iterations
            initial_tour: Starting tour (if None, creates random tour)
            
        Returns:
            Best tour found and its cost
        """
        # Initialize tour
        if initial_tour is None:
            current_tour = list(range(self.n))
            random.shuffle(current_tour)
        else:
            current_tour = initial_tour.copy()
        
        # Initial local search
        current_tour = self.local_search(current_tour)
        best_tour = current_tour.copy()
        best_cost = self.calculate_actual_cost(best_tour)
        
        print(f"Initial cost: {best_cost:.2f}")
        
        for iteration in range(max_iterations):
            # Update penalties based on current local optimum
            self.update_penalties(current_tour)
            
            # Perform local search with updated penalties
            current_tour = self.local_search(current_tour)
            current_cost = self.calculate_actual_cost(current_tour)
            
            # Update best solution
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost
                print(f"Iteration {iteration}: New best cost = {best_cost:.2f}")
        
        print(f"\nFinal best cost: {best_cost:.2f}")
        return best_tour, best_cost


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    # Create a sample distance matrix (80 cities)
    n_cities = 80
    
    # Generate random city coordinates
    coords = np.random.rand(n_cities, 2) * 100
    
    # Calculate distance matrix
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distances[i][j] = np.linalg.norm(coords[i] - coords[j])
    
    # Solve using GLS
    gls = GuidedLocalSearch(distances, lambda_param=0.3)
    best_tour, best_cost = gls.solve(max_iterations=500)
    
    print(f"\nBest tour: {best_tour}")
    print(f"Tour order: {' -> '.join(map(str, best_tour))} -> {best_tour[0]}")