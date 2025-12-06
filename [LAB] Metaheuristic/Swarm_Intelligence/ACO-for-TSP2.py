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

class AntColonyTSP(TSPSolver):

    def __init__(self, distances: np.ndarray, 
                n_ants: int = 10, 
                n_iterations: int = 100, 
                pheromone_weight: float = 1.0, 
                distance_weight: float = 2.0, 
                evaporation_rate: float = 0.5, 
                pheromone_deposit_factor: float = 100):
        """Initialize Ant Colony TSP solver."""
        super().__init__(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromone_weight = pheromone_weight
        self.distance_weight = distance_weight
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit_factor = pheromone_deposit_factor
        
        # Initialize pheromone matrix
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        
    def solve(self, verbose: bool = False) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Ant Colony Optimization.
        """
        start_time = time.time()

        best_tour = None
        best_distance = float('inf')

        for iteration in range(self.n_iterations):
            # if verbose and iteration % 20 == 0:
            #     print(f"\n--- ITERATION {iteration} ---")
        
            # PHASE 1: CONSTRUCTION - Each ant builds a complete tour
            tours = []
            distances = []
            for ant in range(self.n_ants):
                tour = self._construct_tour()
                distance = self.calculate_tour_distance(tour)
                tours.append(tour)
                distances.append(distance)
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour
            #         if verbose:
            #             print(f"  Ant {ant}: NEW BEST = {best_distance:.2f}")
            
            # if verbose and iteration % 20 == 0:
            #     avg_dist = np.mean(distances)
            #     print(f"  Average distance: {avg_dist:.2f}")
            #     print(f"  Best so far: {best_distance:.2f}")
            
            # PHASE 2: PHEROMONE UPDATE - Good tours get reinforced
            self._update_pheromones(tours, distances)
        
        execution_time = time.time() - start_time
        return best_tour, best_distance, execution_time

    def _construct_tour(self) -> List[int]:
        """Construt a tour for one ant"""
        tour = [random.randint(0, self.n_cities - 1)] #Start at random city
        unvisited = set[int](range(self.n_cities)) - {tour[0]}

        while unvisited:
            current_city = tour[-1]
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def _select_next_city(self, current_city:int, unvisited: set[int]) -> int:
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
            prob = (pheromone ** self.pheromone_weight) * ((1.0 / distance) ** self.distance_weight)
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
        self.pheromones *= (1 - self.evaporation_rate)
        
        # Step 2: DEPOSIT - ants leave pheromone on their paths
        for tour, distance in zip(tours, distances):
            # Better tours (shorter distance) deposit more pheromone
            pheromone_deposit = self.pheromone_deposit_factor / distance
            
            # Deposit on all edges in the tour
            for i in range(len(tour)):
                city_a = tour[i]
                city_b = tour[(i + 1) % len(tour)]
                # Symmetric: deposit in both directions
                self.pheromones[city_a][city_b] += pheromone_deposit
                self.pheromones[city_b][city_a] += pheromone_deposit

if __name__ == "__main__":
    n_cities = 8
    np.random.seed(67)
    distances = np.random.randint(10, 100, size=(n_cities, n_cities))
    distances = (distances + distances.T) // 2
    np.fill_diagonal(distances, 0)

    aco_solver = AntColonyTSP(distances, n_ants=10, n_iterations=100)
    tour, distance, time_taken = aco_solver.solve(verbose=True)

    print(f"\n{'='*60}")
    print(f"Final best distance: {distance:.2f}")
    print(f"Final best tour: {tour}")
    print(f"Time taken: {time_taken:.4f} seconds")
    print(f"{'='*60}")