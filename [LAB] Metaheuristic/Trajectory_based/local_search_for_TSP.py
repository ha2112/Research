import random
import math
import time
from typing import List, Tuple

class TSPLocalSearch:
    def __init__(self, cities: List[Tuple[float, float]]):
        """Initialize TSP solver with city coordinates."""
        self.cities = cities
        self.n = len(cities)
        self.dist_matrix = self._compute_distances()
    
    def _compute_distances(self):
        """Precompute distance matrix for efficiency.

        This function calculates the Euclidean distance between
        every pair of cities and stores it in a matrix.
        """
        n = self.n
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                            (self.cities[i][1] - self.cities[j][1])**2)
                dist[i][j] = dist[j][i] = d
        return dist
    
    def tour_cost(self, tour: List[int]) -> float:
        """Calculate total cost (distance) of a tour.

        The tour is assumed to be a list of city indices, visited in order.
        """
        cost = sum(self.dist_matrix[tour[i]][tour[i+1]] 
                   for i in range(len(tour)-1))
        cost += self.dist_matrix[tour[-1]][tour[0]] # complete cycle by returning to start
        return cost
    
    def generate_initial_solution(self, method='random') -> List[int]:
        """Generate initial tour using specified method.

        Parameters:
            method: 'random' shuffles cities, 'nearest' starts greedy from city 0.

        Returns:
            List of city indices as the initial tour.
        """
        if method == 'random':
            tour = list(range(self.n))
            random.shuffle(tour)
            return tour
        elif method == 'nearest':
            return self._nearest_neighbor()
        return list(range(self.n))
    
    def _nearest_neighbor(self) -> List[int]:
        """Construct tour using nearest neighbor heuristic.

        Start from the first city (index 0); at each step, 
        select the closest unvisited city.
        """
        tour = [0]
        unvisited = set(range(1, self.n))
        
        while unvisited:
            curr = tour[-1]
            # Find the nearest unvisited city to the current city
            nearest = min(unvisited, key=lambda x: self.dist_matrix[curr][x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        return tour
    
    def two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Perform 2-opt swap: reverse segment between i and j.

        This operation breaks two edges and reconnects them in a new way
        to create a different tour that may have a lower cost.
        """
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
     
        
        return best_tour, best_cost
    
    def three_opt_improvement(self, tour: List[int], max_iter=500) -> Tuple[List[int], float]:
        """Apply 3-opt local search (simplified version).

        Considers breaking the tour at 3 positions and joining the segments
        in different orders to (potentially) reduce the total tour cost.
        This function only explores a subset of 3-opt moves for simplicity.
        """
        best_tour = tour[:]
        best_cost = self.tour_cost(best_tour)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            for i in range(self.n - 2):
                for j in range(i + 2, self.n - 1):
                    for k in range(j + 2, self.n):
                        # Generate two possible new tours by reconnecting segments in different ways
                        tours_to_try = [
                            best_tour[:i+1] + best_tour[j+1:k+1] + best_tour[i+1:j+1] + best_tour[k+1:],
                            best_tour[:i+1] + best_tour[j+1:k+1][::-1] + best_tour[i+1:j+1] + best_tour[k+1:],
                        ]
                        
                        for new_tour in tours_to_try:
                            new_cost = self.tour_cost(new_tour)
                            # If cost is improved, accept the new tour and break from the loop to continue searching
                            if new_cost < best_cost:
                                best_tour = new_tour
                                best_cost = new_cost
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return best_tour, best_cost
    
    def solve(self, method='2opt', initial='nearest', restarts=5):
        """
        Solve TSP using local search with multiple restarts.
        
        Args:
            method: '2opt' or '3opt'
            initial: 'random' or 'nearest' for initial solution
            restarts: number of random restarts
        Returns:
            The best tour found and its associated cost.
        """
        best_tour = None
        best_cost = float('inf')
        
        print(f"Solving TSP with {self.n} cities using {method} local search...")
        print(f"Running {restarts} restarts...\n")
        
        for restart in range(restarts):
            # Generate initial solution: first restart uses nearest neighbor,
            # others use random initialization for diversification
            if restart == 0 and initial == 'nearest':
                tour = self.generate_initial_solution('nearest')
            else:
                tour = self.generate_initial_solution('random')
            
            initial_cost = self.tour_cost(tour)
            
            # Apply chosen local search method to improve solution
            start_time = time.time()
            if method == '2opt':
                tour, cost = self.two_opt_improvement(tour)
            elif method == '3opt':
                tour, cost = self.three_opt_improvement(tour)
            else:
                # fallback: no local search, return initial cost
                cost = initial_cost
            
            elapsed = time.time() - start_time
            
            improvement = ((initial_cost - cost) / initial_cost) * 100
            print(f"Restart {restart + 1}: Initial={initial_cost:.2f}, "
                  f"Final={cost:.2f}, Improvement={improvement:.1f}%, Time={elapsed:.3f}s")
            
            # Keep the best overall solution found so far
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        print(f"\nBest solution found: {best_cost:.2f}")
        return best_tour, best_cost


# Example usage: run as a script for demo
if __name__ == "__main__":
    # Generate random cities (2D coordinates in a 100x100 square)
    random.seed(42)
    num_cities = 50
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) 
              for _ in range(num_cities)]
    
    # Create solver and run 2-opt and 3-opt local search
    solver = TSPLocalSearch(cities)
    
    print("=" * 60)
    print("2-OPT LOCAL SEARCH")
    print("=" * 60)
    best_tour_2opt, best_cost_2opt = solver.solve(method='2opt', restarts=3)
    
    print("\n" + "=" * 60)
    print("3-OPT LOCAL SEARCH")
    print("=" * 60)
    best_tour_3opt, best_cost_3opt = solver.solve(method='3opt', restarts=2)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"2-opt best: {best_cost_2opt:.2f}")
    print(f"3-opt best: {best_cost_3opt:.2f}")
    print(f"Difference: {abs(best_cost_2opt - best_cost_3opt):.2f}")