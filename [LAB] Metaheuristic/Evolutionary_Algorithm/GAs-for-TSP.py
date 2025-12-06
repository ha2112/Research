import numpy as np
import random
import time
from typing import List, Tuple, Optional

# ============================================================================
# SHARED COMPONENTS
# ============================================================================

def generate_cities(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate n random 2D coordinates representing cities.
    Cities are placed inside a 100x100 grid.
    """
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100  # Cities in 100x100 grid

def calculate_total_distance(tour: List[int], cities: np.ndarray) -> float:
    """
    Compute the total path length of a tour visiting all cities in order and returning to the start.
    Uses Euclidean distance between each pair of consecutive cities.
    """
    distance = 0.0
    for i in range(len(tour)):
        city_a = cities[tour[i]]
        city_b = cities[tour[(i + 1) % len(tour)]]  # Wrap around to start
        distance += np.linalg.norm(city_a - city_b)
    return distance

# ============================================================================
# BACKTRACKING ALGORITHM (EXACT SOLUTION)
# ============================================================================

def backtracking_tsp(cities: np.ndarray, 
                     max_time: float = 60.0) -> Tuple[Optional[List[int]], float, bool, int]:
    """
    Solve TSP using backtracking with branch-and-bound optimization.
    
    This algorithm explores all possible permutations but prunes branches
    that cannot lead to better solutions than the current best.
    
    Args:
        cities: array of city coordinates
        max_time: maximum execution time in seconds (prevents infinite runs)
        
    Returns:
        (best_tour, best_distance, completed, nodes_explored)
        - best_tour: best solution found (None if timeout before first solution)
        - best_distance: distance of best tour
        - completed: True if search completed, False if timed out
        - nodes_explored: number of search tree nodes explored
    """
    n_cities = len(cities)
    best_tour = None
    best_distance = float('inf')
    nodes_explored = 0
    start_time = time.time()
    
    # Precompute distance matrix for efficiency
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    def backtrack(path: List[int], visited: set, current_distance: float):
        nonlocal best_tour, best_distance, nodes_explored
        
        nodes_explored += 1
        
        # Check timeout
        if time.time() - start_time > max_time:
            return False  # Signal timeout
        
        # Base case: all cities visited
        if len(path) == n_cities:
            # Complete the tour by returning to start
            total_distance = current_distance + dist_matrix[path[-1]][path[0]]
            if total_distance < best_distance:
                best_distance = total_distance
                best_tour = path.copy()
            return True
        
        # Pruning: if current partial path is already worse than best, stop
        if current_distance >= best_distance:
            return True
        
        # Try each unvisited city
        current_city = path[-1]
        for next_city in range(n_cities):
            if next_city not in visited:
                # Add city to path
                path.append(next_city)
                visited.add(next_city)
                new_distance = current_distance + dist_matrix[current_city][next_city]
                
                # Recurse
                if not backtrack(path, visited, new_distance):
                    return False  # Timeout propagation
                
                # Backtrack
                path.pop()
                visited.remove(next_city)
        
        return True
    
    # Start from city 0 (arbitrary choice due to symmetry)
    completed = backtrack([0], {0}, 0.0)
    
    return best_tour, best_distance, completed, nodes_explored

# ============================================================================
# GENETIC ALGORITHM WITH INTEGER ENCODING (PERMUTATION-BASED)
# ============================================================================

def fitness_integer(tour: List[int], cities: np.ndarray) -> float:
    """
    Fitness function for permutation representation.
    Returns the inverse of tour distance as fitness (higher is better).
    """
    distance = calculate_total_distance(tour, cities)
    return 1.0 / distance if distance > 0 else 0

def tournament_selection(population: List[List[int]], 
                        fitness_scores: List[float], 
                        tournament_size: int = 3) -> List[int]:
    """
    Parent selection via tournament selection.
    Randomly picks a group of tournament_size candidates from the population,
    then selects and returns the fittest one.
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx].copy()

def ordered_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Ordered Crossover (OX1) operator for permutations (for TSP).
    
    Steps:
    1. Select a random substring of consecutive indices from parent1.
    2. Copy that substring to the same position in the child.
    3. Fill remaining positions with cities from parent2 in the order they appear, skipping any already present.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    # Step 1: Copy substring from parent1 into child
    child = [-1] * size
    child[start:end] = parent1[start:end]
    
    # Step 2,3: Fill all other positions from parent2, in order, skipping already used cities
    current_pos = end
    for city in parent2[end:] + parent2[:end]:  # Circular iteration through parent2
        if city not in child:
            if current_pos >= size:
                current_pos = 0
            child[current_pos] = city
            current_pos += 1
    
    return child

def swap_mutation(tour: List[int], mutation_rate: float) -> List[int]:
    """
    Swap mutation for permutation encoding.
    With probability mutation_rate, selects two random cities and swaps them in the tour.
    """
    tour = tour.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def ga_integer_encoding(cities: np.ndarray, 
                       pop_size: int = 100, 
                       generations: int = 500, 
                       mutation_rate: float = 0.2) -> Tuple[List[int], float, float]:
    """
    Genetic Algorithm main loop for permutation/integer encoding (order-based, for TSP).
    
    Args:
        cities: array of city coordinates
        pop_size: number of individuals in the population
        generations: number of generations to run
        mutation_rate: probability of mutation per individual per generation
        
    Returns:
        (best_tour, best_distance, elapsed_time): best solution found, its distance, and runtime
    """
    start_time = time.time()
    n_cities = len(cities)
    
    # Initialize population with random permutations of city indices
    population = [random.sample(range(n_cities), n_cities) for _ in range(pop_size)]
    
    best_tour = None
    best_distance = float('inf')
    
    for generation in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [fitness_integer(tour, cities) for tour in population]
        
        # Track best solution in this generation
        gen_best_idx = fitness_scores.index(max(fitness_scores))
        gen_best_distance = calculate_total_distance(population[gen_best_idx], cities)
        
        if gen_best_distance < best_distance:
            best_distance = gen_best_distance
            best_tour = population[gen_best_idx].copy()
        
        # Create new population (with elitism)
        new_population = []
        
        # Elitism: copy best individual to next generation
        new_population.append(population[gen_best_idx].copy())
        
        # Fill the rest of the new population
        while len(new_population) < pop_size:
            # Parent selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover to produce a child
            child = ordered_crossover(parent1, parent2)
            
            # Mutation
            child = swap_mutation(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population  # Continue with new population
    
    elapsed_time = time.time() - start_time
    return best_tour, best_distance, elapsed_time

# ============================================================================
# GENETIC ALGORITHM WITH FLOAT ENCODING (RANDOM KEYS)
# ============================================================================

def decode_float_chromosome(chromosome: List[float]) -> List[int]:
    """
    Convert a float vector (chromosome) to a TSP permutation using the random keys scheme.
    The permutation is defined by sorting the chromosome and using the index order.
    
    Example:
        [0.81, 0.12, 0.95, 0.34] -> [1, 3, 0, 2]
        (Order is: 0.12->idx1, 0.34->idx3, 0.81->idx0, 0.95->idx2)
    """
    indexed_chromosome = list(enumerate(chromosome))
    indexed_chromosome.sort(key=lambda x: x[1])
    return [idx for idx, _ in indexed_chromosome]

def fitness_float(chromosome: List[float], cities: np.ndarray) -> float:
    """
    Fitness function for float encoding (random keys encoding).
    Computes the total distance after mapping chromosome to a city permutation.
    """
    tour = decode_float_chromosome(chromosome)
    distance = calculate_total_distance(tour, cities)
    return 1.0 / distance if distance > 0 else 0

def tournament_selection_float(population: List[List[float]], 
                               fitness_scores: List[float], 
                               tournament_size: int = 3) -> List[float]:
    """
    Tournament selection for float-encoded population.
    Randomly sample tournament_size individuals and return a copy of the fittest one.
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx].copy()

def uniform_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Uniform crossover for float chromosomes (random keys).
    For each gene position, randomly select from parent1 or parent2 (50% each).
    """
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        child.append(gene1 if random.random() < 0.5 else gene2)
    return child

def random_reset_mutation(chromosome: List[float], mutation_rate: float) -> List[float]:
    """
    Random-reset mutation for float encoding.
    For each gene, with probability mutation_rate, replace it with a random float in [0,1).
    """
    chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.random()
    return chromosome

def ga_float_encoding(cities: np.ndarray, 
                     pop_size: int = 100, 
                     generations: int = 500, 
                     mutation_rate: float = 0.2) -> Tuple[List[int], float, float]:
    """
    Genetic Algorithm main loop for float/random-keys encoding.
    
    Args:
        cities: array of city coordinates
        pop_size: number of individuals in the population
        generations: number of generations to run
        mutation_rate: probability of mutation per gene per generation
        
    Returns:
        (best_tour, best_distance, elapsed_time): best solution found, its distance, and runtime
    """
    start_time = time.time()
    n_cities = len(cities)
    
    # Initialize population with random float vectors, one per individual
    population = [[random.random() for _ in range(n_cities)] for _ in range(pop_size)]
    
    best_tour = None
    best_distance = float('inf')
    
    for generation in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [fitness_float(chromosome, cities) for chromosome in population]
        
        # Track best solution in this generation
        gen_best_idx = fitness_scores.index(max(fitness_scores))
        gen_best_tour = decode_float_chromosome(population[gen_best_idx])
        gen_best_distance = calculate_total_distance(gen_best_tour, cities)
        
        if gen_best_distance < best_distance:
            best_distance = gen_best_distance
            best_tour = gen_best_tour
        
        # Create new population (with elitism)
        new_population = []
        
        # Elitism: copy best individual to the next generation
        new_population.append(population[gen_best_idx].copy())
        
        # Fill the rest of the new population
        while len(new_population) < pop_size:
            # Parent selection
            parent1 = tournament_selection_float(population, fitness_scores)
            parent2 = tournament_selection_float(population, fitness_scores)
            
            # Uniform crossover for float chromosomes
            child = uniform_crossover(parent1, parent2)
            
            # Mutation
            child = random_reset_mutation(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population  # Continue with new population
    
    elapsed_time = time.time() - start_time
    return best_tour, best_distance, elapsed_time

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print header and setup problem
    print("=" * 70)
    print("TRAVELING SALESPERSON PROBLEM - ALGORITHM COMPARISON")
    print("=" * 70)
    
    # Problem configuration
    n_cities = 13  # Cities to generate
    cities = generate_cities(n_cities)
    
    print(f"\nProblem: {n_cities} cities")
    print(f"GA Parameters: Pop=100, Gen=500, Mutation=0.2")
    print(f"Backtracking: Max time=60s")
    
    # Run Backtracking Algorithm (Exact Solution)
    print("\n" + "-" * 70)
    print("RUNNING: Backtracking Algorithm (Exact Solution)")
    print("-" * 70)
    
    bt_start = time.time()
    bt_tour, bt_distance, bt_completed, bt_nodes = backtracking_tsp(cities, max_time=60.0)
    bt_time = time.time() - bt_start
    
    if bt_tour:
        print(f"Best tour found: {bt_tour}")
        print(f"Total distance: {bt_distance:.2f}")
        print(f"Execution time: {bt_time:.2f}s")
        print(f"Nodes explored: {bt_nodes:,}")
        print(f"Status: {'✓ COMPLETE' if bt_completed else '⚠ TIMEOUT'}")
    else:
        print("No solution found within time limit")
    
    # Run Integer Encoding GA (Permutation based)
    print("\n" + "-" * 70)
    print("RUNNING: Integer Encoding GA (Permutation-Based)")
    print("-" * 70)
    
    int_tour, int_distance, int_time = ga_integer_encoding(
        cities, 
        pop_size=100, 
        generations=500, 
        mutation_rate=0.2
    )
    
    print(f"Best tour found: {int_tour}")
    print(f"Total distance: {int_distance:.2f}")
    print(f"Execution time: {int_time:.2f}s")
    
    # Run Float Encoding GA (Random keys)
    print("\n" + "-" * 70)
    print("RUNNING: Float Encoding GA (Random Keys)")
    print("-" * 70)
    
    float_tour, float_distance, float_time = ga_float_encoding(
        cities, 
        pop_size=100, 
        generations=500, 
        mutation_rate=0.2
    )
    
    print(f"Best tour found: {float_tour}")
    print(f"Total distance: {float_distance:.2f}")
    print(f"Execution time: {float_time:.2f}s")
    
    # Compare all approaches
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Solution Quality Comparison
    print("\n📊 SOLUTION QUALITY:")
    print("-" * 70)
    if bt_tour:
        print(f"Backtracking:      {bt_distance:.2f} {'(OPTIMAL)' if bt_completed else '(BEST FOUND)'}")
    print(f"Integer Encoding:  {int_distance:.2f}")
    print(f"Float Encoding:    {float_distance:.2f}")
    
    if bt_tour and bt_completed:
        int_error = ((int_distance - bt_distance) / bt_distance) * 100
        float_error = ((float_distance - bt_distance) / bt_distance) * 100
        print(f"\nError vs Optimal:")
        print(f"  Integer GA: +{int_error:.1f}%")
        print(f"  Float GA:   +{float_error:.1f}%")
    
    # Runtime Comparison
    print(f"\n⏱️  EXECUTION TIME:")
    print("-" * 70)
    if bt_tour:
        print(f"Backtracking:      {bt_time:.2f}s")
    print(f"Integer Encoding:  {int_time:.2f}s")
    print(f"Float Encoding:    {float_time:.2f}s")
    
    if bt_tour:
        print(f"\nSpeedup vs Backtracking:")
        print(f"  Integer GA: {bt_time/int_time:.1f}x faster")
        print(f"  Float GA:   {bt_time/float_time:.1f}x faster")
    
    # Winner Determination
    print(f"\n🏆 BEST SOLUTION:")
    print("-" * 70)
    
    best_dist = min(int_distance, float_distance)
    if bt_tour and bt_distance < best_dist:
        print(f"Backtracking found the best solution: {bt_distance:.2f}")
        print(f"(But took {bt_time:.2f}s vs ~{(int_time + float_time)/2:.2f}s for GAs)")
    elif int_distance < float_distance:
        improvement = ((float_distance - int_distance) / float_distance) * 100
        print(f"✓ Integer Encoding GA ({improvement:.1f}% better than Float)")
    elif float_distance < int_distance:
        improvement = ((int_distance - float_distance) / int_distance) * 100
        print(f"✓ Float Encoding GA ({improvement:.1f}% better than Integer)")
    else:
        print("= Both GAs found equal quality solutions")
    
    # Scalability Note
    print(f"\n💡 SCALABILITY INSIGHTS:")
    print("-" * 70)
    print(f"For {n_cities} cities:")
    print(f"  - Backtracking explored {bt_nodes:,} nodes")
    print(f"  - GAs evaluated ~{100 * 500:,} solutions each")
    print(f"\nFor larger problems (20+ cities):")
    print(f"  - Backtracking becomes impractical (factorial growth)")
    print(f"  - GAs remain efficient with consistent runtime")
    print(f"  - GAs are the practical choice for real-world TSP")