import numpy as np
import time
from typing import List, Tuple
import random

class TSPSolver:
    def __init__(self, distances: np.ndarray):
        """
        Initialize TSP solver with distance matrix.
        
        Args:
            distances: Square matrix where distances[i][j] is distance from city i to j
        """
        self.distances = distances
        self.n_cities = len(distances)
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour."""
        distance = sum(self.distances[tour[i]][tour[i+1]] 
                      for i in range(len(tour)-1))
        distance += self.distances[tour[-1]][tour[0]]  # Return to start
        return distance


class BacktrackingTSP(TSPSolver):
    """Exact solution using backtracking - guarantees optimal solution."""
    
    def solve(self) -> Tuple[List[int], float, float]:
        """
        Solve TSP using backtracking.
        
        Returns:
            tuple: (best_tour, best_distance, execution_time)
        """
        start_time = time.time()
        
        self.best_distance = float('inf')
        self.best_tour = None
        
        # Start from city 0
        current_tour = [0]
        visited = {0}
        current_distance = 0
        
        self._backtrack(current_tour, visited, current_distance)
        
        execution_time = time.time() - start_time
        return self.best_tour, self.best_distance, execution_time
    
    def _backtrack(self, tour: List[int], visited: set, current_distance: float):
        """Recursive backtracking function."""
        # If all cities visited, check if this is best solution
        if len(tour) == self.n_cities:
            # Add distance back to start
            total_distance = current_distance + self.distances[tour[-1]][tour[0]]
            if total_distance < self.best_distance:
                self.best_distance = total_distance
                self.best_tour = tour.copy()
            return
        
        # Pruning: if current distance already exceeds best, stop exploring
        if current_distance >= self.best_distance:
            return
        
        # Try adding each unvisited city
        current_city = tour[-1]
        for next_city in range(self.n_cities):
            if next_city not in visited:
                # Add city to tour
                tour.append(next_city)
                visited.add(next_city)
                new_distance = current_distance + self.distances[current_city][next_city]
                
                # Recurse
                self._backtrack(tour, visited, new_distance)
                
                # Backtrack
                tour.pop()
                visited.remove(next_city)


class AntColonyTSP(TSPSolver):
    """Approximate solution using Ant Colony Optimization - fast heuristic."""
    
    def __init__(self, distances: np.ndarray, n_ants: int = 10, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.5,
                 q: float = 100):
        """
        Initialize ACO solver.
        
        Args:
            distances: Distance matrix
            n_ants: Number of ants per iteration
            n_iterations: Number of iterations to run
            alpha: Pheromone importance factor
            beta: Distance importance factor
            evaporation: Pheromone evaporation rate (0-1)
            q: Pheromone deposit factor
        """
        super().__init__(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        
        # Initialize pheromone matrix
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        
    def solve(self, verbose: bool = False) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Ant Colony Optimization.
        
        Each iteration consists of 3 main phases:
        1. CONSTRUCTION: All ants build complete tours
        2. EVALUATION: Calculate distance of each tour
        3. PHEROMONE UPDATE: Evaporate old pheromones and deposit new ones
        
        Returns:
            tuple: (best_tour, best_distance, execution_time)
        """
        start_time = time.time()
        
        best_tour = None
        best_distance = float('inf')
        
        for iteration in range(self.n_iterations):
            if verbose and iteration % 20 == 0:
                print(f"\n--- ITERATION {iteration} ---")
            
            # PHASE 1: CONSTRUCTION - Each ant builds a complete tour
            tours = []
            distances = []
            
            for ant in range(self.n_ants):
                tour = self._construct_tour()  # Ant walks step-by-step
                distance = self.calculate_tour_distance(tour)
                tours.append(tour)
                distances.append(distance)
                
                # Update best solution
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour
                    if verbose:
                        print(f"  Ant {ant}: NEW BEST = {best_distance:.2f}")
            
            if verbose and iteration % 20 == 0:
                avg_dist = np.mean(distances)
                print(f"  Average distance: {avg_dist:.2f}")
                print(f"  Best so far: {best_distance:.2f}")
            
            # PHASE 2: PHEROMONE UPDATE - Good tours get reinforced
            self._update_pheromones(tours, distances)
        
        execution_time = time.time() - start_time
        return best_tour, best_distance, execution_time
    
    def _construct_tour(self) -> List[int]:
        """
        Construct a tour for one ant - this is ONE ANT'S JOURNEY.
        
        The ant builds a tour step-by-step:
        - Step 1: Start at a random city
        - Step 2-N: At each step, choose next unvisited city based on:
            * Pheromone strength (learning from past ants)
            * Distance (prefer closer cities)
        - This is probabilistic, not deterministic!
        """
        tour = [random.randint(0, self.n_cities - 1)]  # Start at random city
        unvisited = set(range(self.n_cities)) - {tour[0]}
        
        # Each ant makes (n_cities - 1) decisions
        while unvisited:
            current_city = tour[-1]
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour
    
    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        """
        Select next city based on pheromone and distance.
        
        This is the CORE DECISION MECHANISM - happens at each step for each ant.
        
        The probability of choosing city j from city i is:
        P(i→j) = [τ(i,j)^α * η(i,j)^β] / Σ[τ(i,k)^α * η(i,k)^β] for all k in unvisited
        
        Where:
        - τ(i,j) = pheromone on edge i→j (higher = more ants used this)
        - η(i,j) = 1/distance(i,j) (visibility, higher = closer city)
        - α = pheromone importance weight
        - β = distance importance weight
        """
        probabilities = []
        
        for city in unvisited:
            # Avoid division by zero
            distance = max(self.distances[current_city][city], 0.0001)
            pheromone = self.pheromones[current_city][city]
            
            # Probability based on pheromone^alpha * (1/distance)^beta
            prob = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
            probabilities.append(prob)
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        # Select city based on probabilities (STOCHASTIC choice)
        unvisited_list = list(unvisited)
        next_city = np.random.choice(unvisited_list, p=probabilities)
        
        return next_city
    
    def _update_pheromones(self, tours: List[List[int]], distances: List[float]):
        """
        Update pheromone matrix after all ants complete tours.
        
        This happens ONCE per iteration after ALL ants finish.
        
        Two sub-steps:
        1. EVAPORATION: Reduce all pheromones by evaporation rate
           - This "forgets" old information gradually
           - τ(i,j) = τ(i,j) * (1 - ρ) where ρ = evaporation rate
        
        2. DEPOSIT: Each ant deposits pheromone on its path
           - Better tours (shorter) deposit MORE pheromone
           - Δτ(i,j) = Q / L where Q = constant, L = tour length
           - Shorter tours → larger Δτ → stronger reinforcement
        """
        # Step 1: EVAPORATION - all pheromones decay
        self.pheromones *= (1 - self.evaporation)
        
        # Step 2: DEPOSIT - ants leave pheromone on their paths
        for tour, distance in zip(tours, distances):
            # Better tours (shorter distance) deposit more pheromone
            pheromone_deposit = self.q / distance
            
            # Deposit on all edges in the tour
            for i in range(len(tour)):
                city_a = tour[i]
                city_b = tour[(i + 1) % len(tour)]
                # Symmetric: deposit in both directions
                self.pheromones[city_a][city_b] += pheromone_deposit
                self.pheromones[city_b][city_a] += pheromone_deposit


# Example usage and comparison
def compare_algorithms():
    """Compare ACO and Backtracking on a sample TSP problem."""
    
    # Create a small random distance matrix (symmetric)
    n_cities = 8
    np.random.seed(42)
    distances = np.random.randint(10, 100, size=(n_cities, n_cities))
    distances = (distances + distances.T) // 2  # Make symmetric
    np.fill_diagonal(distances, 0)  # Zero distance to self
    
    print(f"Solving TSP for {n_cities} cities\n")
    print("Distance Matrix:")
    print(distances)
    print("\n" + "="*60 + "\n")
    
    # Solve with Backtracking (exact solution)
    print("BACKTRACKING (Exact Algorithm):")
    bt_solver = BacktrackingTSP(distances)
    bt_tour, bt_distance, bt_time = bt_solver.solve()
    print(f"Best tour: {bt_tour}")
    print(f"Distance: {bt_distance:.2f}")
    print(f"Time: {bt_time:.4f} seconds")
    print("\n" + "="*60 + "\n")
    
    # Solve with ACO (heuristic)
    print("ANT COLONY OPTIMIZATION (Heuristic Algorithm):")
    aco_solver = AntColonyTSP(distances, n_ants=20, n_iterations=100)
    
    # Set verbose=True to see the learning process in action!
    aco_tour, aco_distance, aco_time = aco_solver.solve(verbose=False)
    
    print(f"Best tour: {aco_tour}")
    print(f"Distance: {aco_distance:.2f}")
    print(f"Time: {aco_time:.4f} seconds")
    print("\n" + "="*60 + "\n")
    
    # Comparison
    print("COMPARISON:")
    print(f"Backtracking distance: {bt_distance:.2f} (optimal)")
    print(f"ACO distance: {aco_distance:.2f}")
    print(f"ACO accuracy: {(bt_distance/aco_distance)*100:.2f}% of optimal")
    print(f"\nBacktracking time: {bt_time:.4f}s")
    print(f"ACO time: {aco_time:.4f}s")
    print(f"Speed ratio: {bt_time/aco_time:.2f}x")
    print("\n" + "="*60)
    
    print("\nKEY DIFFERENCES:")
    print("• Backtracking: Guarantees optimal solution but exponential time O(n!)")
    print("• ACO: Fast approximation, scales to large problems, may not be optimal")
    print("• For n>15 cities, backtracking becomes impractical")
    print("• ACO can handle 100+ cities efficiently")
    
    print("\n" + "="*60)
    print("\nACO ITERATION BREAKDOWN:")
    print("="*60)
    print("\nEach iteration has 3 phases:\n")
    print("1. CONSTRUCTION PHASE (parallel):")
    print("   - Each of the 20 ants builds a complete tour")
    print("   - Each ant makes step-by-step decisions:")
    print("     * Start at random city")
    print("     * For each unvisited city, calculate probability based on:")
    print("       - Pheromone strength (what previous ants found good)")
    print("       - Distance (prefer closer cities)")
    print("     * Randomly select next city using these probabilities")
    print("     * Repeat until all cities visited")
    print("   - Result: 20 different tours\n")
    print("2. EVALUATION PHASE:")
    print("   - Calculate total distance for each of the 20 tours")
    print("   - Track the best tour found so far\n")
    print("3. PHEROMONE UPDATE PHASE:")
    print("   - EVAPORATE: Reduce all pheromones by evaporation rate")
    print("     (prevents convergence to suboptimal solutions)")
    print("   - DEPOSIT: Each ant deposits pheromone on its path")
    print("     * Better tours deposit MORE pheromone")
    print("     * This reinforces good paths for future iterations")
    print("\nThis cycle repeats for 100 iterations, allowing the")
    print("colony to collectively learn good solutions!")


if __name__ == "__main__":
    compare_algorithms()
    
    # Try verbose mode to see ACO learning in real-time!
    print("\n\n" + "="*60)
    print("RUNNING ACO IN VERBOSE MODE")
    print("="*60)
    print("Watch how the ants learn and improve over iterations:\n")
    
    n_cities = 8
    np.random.seed(42)
    distances = np.random.randint(10, 100, size=(n_cities, n_cities))
    distances = (distances + distances.T) // 2
    np.fill_diagonal(distances, 0)
    
    aco_verbose = AntColonyTSP(distances, n_ants=10, n_iterations=50)
    tour, dist, time_taken = aco_verbose.solve(verbose=True)
    
    print(f"\n{'='*60}")
    print(f"Final best distance: {dist:.2f}")
    print(f"Final best tour: {tour}")