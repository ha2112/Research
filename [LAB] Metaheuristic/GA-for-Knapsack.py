"""
0/1 KNAPSACK PROBLEM SOLVER
Comparing Genetic Algorithm vs Dynamic Programming

This script implements both approaches and provides comprehensive analysis
of their performance, solution quality, and trade-offs.
"""

import random
import time
import numpy as np
from typing import List, Tuple

# ============================================================================
# PROBLEM DATA
# ============================================================================
WEIGHTS = [12, 7, 11, 8, 9, 13, 5, 6, 10, 8, 9, 11, 14, 7, 6, 10, 12, 5, 8, 9]
VALUES = [25, 18, 22, 19, 21, 28, 15, 16, 23, 17, 20, 24, 30, 16, 15, 25, 27, 14, 19, 21]
CAPACITY = 100
N_ITEMS = len(WEIGHTS)

# ============================================================================
# GA PARAMETERS
# ============================================================================
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5


# ============================================================================
# GENETIC ALGORITHM IMPLEMENTATION
# ============================================================================

def fitness(chromosome: List[int]) -> int:
    """
    Calculate fitness of a chromosome (binary string).
    Returns total value if weight constraint satisfied, else 0.
    
    Analysis: O(n) time complexity for each fitness evaluation
    """
    total_weight = sum(WEIGHTS[i] for i in range(N_ITEMS) if chromosome[i] == 1)
    total_value = sum(VALUES[i] for i in range(N_ITEMS) if chromosome[i] == 1)
    
    # Penalty approach: invalid solutions get 0 fitness
    if total_weight > CAPACITY:
        return 0
    return total_value


def get_weight(chromosome: List[int]) -> int:
    """Calculate total weight of items in chromosome."""
    return sum(WEIGHTS[i] for i in range(N_ITEMS) if chromosome[i] == 1)


def initialize_population(size: int) -> List[List[int]]:
    """
    Create initial population of random chromosomes.
    Each chromosome is a binary list of length N_ITEMS.
    
    Analysis: O(size * n) time complexity
    """
    population = []
    for _ in range(size):
        chromosome = [random.randint(0, 1) for _ in range(N_ITEMS)]
        population.append(chromosome)
    return population


def tournament_selection(population: List[List[int]], fitnesses: List[int]) -> List[int]:
    """
    Select a parent using tournament selection.
    Randomly pick TOURNAMENT_SIZE individuals and return the best.
    
    Analysis: O(TOURNAMENT_SIZE) time complexity per selection
    """
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index]


def single_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Perform single-point crossover between two parents.
    Returns two offspring.
    
    Analysis: O(n) time complexity
    """
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    
    point = random.randint(1, N_ITEMS - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    
    return offspring1, offspring2


def bit_flip_mutation(chromosome: List[int]) -> List[int]:
    """
    Perform bit flip mutation on a chromosome.
    Each bit has MUTATION_RATE probability of flipping.
    
    Analysis: O(n) time complexity
    """
    mutated = chromosome.copy()
    for i in range(N_ITEMS):
        if random.random() < MUTATION_RATE:
            mutated[i] = 1 - mutated[i]
    return mutated


def genetic_algorithm() -> Tuple[List[int], int, int, float, List[int]]:
    """
    Main GA loop to solve 0/1 Knapsack problem.
    
    Returns: (best_chromosome, best_fitness, best_generation, execution_time, fitness_history)
    
    Analysis:
    - Overall Time Complexity: O(G * P * n)
      where G = generations, P = population size, n = number of items
    - Space Complexity: O(P * n) for population storage
    """
    start_time = time.time()
    
    # Initialize population
    population = initialize_population(POPULATION_SIZE)
    
    # Track best solution and fitness history
    best_chromosome = None
    best_fitness = 0
    best_generation = 0
    fitness_history = []
    
    # Evolution loop
    for generation in range(GENERATIONS):
        # Evaluate fitness for all chromosomes - O(P * n)
        fitnesses = [fitness(chrom) for chrom in population]
        
        # Track statistics
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        fitness_history.append(max_fitness)
        
        # Update best solution
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[fitnesses.index(max_fitness)].copy()
            best_generation = generation
        
        # Create next generation
        new_population = []
        
        # Elitism: preserve best solution
        new_population.append(best_chromosome.copy())
        
        # Generate offspring - O(P * n) per generation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            
            offspring1 = bit_flip_mutation(offspring1)
            offspring2 = bit_flip_mutation(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(offspring2)
        
        population = new_population
        
        # Progress reporting
        if (generation + 1) % 50 == 0:
            print(f"  Generation {generation + 1:3d}: Best={best_fitness}, Avg={avg_fitness:.2f}")
    
    execution_time = time.time() - start_time
    
    return best_chromosome, best_fitness, best_generation, execution_time, fitness_history


# ============================================================================
# DYNAMIC PROGRAMMING IMPLEMENTATION
# ============================================================================

def knapsack_dp() -> Tuple[int, int, List[int], float]:
    """
    Solve 0/1 Knapsack problem using Dynamic Programming.
    
    Returns: (max_value, total_weight, selected_items, execution_time)
    
    Recurrence Relation:
    dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])
    
    Analysis:
    - Time Complexity: O(n * W) - Pseudo-polynomial
      * Polynomial in numeric value of W, not its bit representation
      * W is represented in log(W) bits, so actually exponential in input size
    - Space Complexity: O(n * W) for the DP table
      * Can be optimized to O(W) using rolling array
    """
    start_time = time.time()
    
    # Create DP table: (n+1) x (capacity+1)
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0 for _ in range(CAPACITY + 1)] for _ in range(N_ITEMS + 1)]
    
    # Fill the DP table - O(n * W)
    for i in range(1, N_ITEMS + 1):
        for w in range(CAPACITY + 1):
            # Option 1: Don't take item i-1
            dont_take = dp[i-1][w]
            
            # Option 2: Take item i-1 (if it fits)
            take = 0
            if WEIGHTS[i-1] <= w:
                take = dp[i-1][w - WEIGHTS[i-1]] + VALUES[i-1]
            
            dp[i][w] = max(dont_take, take)
    
    # Maximum value
    max_value = dp[N_ITEMS][CAPACITY]
    
    # Backtrack to find selected items - O(n)
    selected_items = []
    w = CAPACITY
    
    for i in range(N_ITEMS, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= WEIGHTS[i-1]
    
    selected_items.reverse()
    
    # Calculate total weight
    total_weight = sum(WEIGHTS[i] for i in selected_items)
    
    execution_time = time.time() - start_time
    
    return max_value, total_weight, selected_items, execution_time


# ============================================================================
# ANALYSIS AND COMPARISON
# ============================================================================

def compare_solutions(ga_solution: List[int], ga_value: int, ga_time: float,
                     dp_value: int, dp_items: List[int], dp_time: float) -> None:
    """
    Perform detailed comparison and analysis of both solutions.
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Solution Quality
    print("\n1. SOLUTION QUALITY")
    print("-" * 80)
    print(f"DP Optimal Value:     {dp_value}")
    print(f"GA Approximate Value: {ga_value}")
    
    if ga_value == dp_value:
        print("✓ GA found OPTIMAL solution!")
        optimality_gap = 0.0
    else:
        optimality_gap = ((dp_value - ga_value) / dp_value) * 100
        print(f"✗ GA solution is suboptimal")
        print(f"  Optimality Gap: {optimality_gap:.2f}%")
        print(f"  Missing Value: {dp_value - ga_value}")
    
    # Execution Time
    print("\n2. EXECUTION TIME")
    print("-" * 80)
    print(f"DP Time:  {dp_time*1000:.4f} ms")
    print(f"GA Time:  {ga_time*1000:.4f} ms")
    speedup = ga_time / dp_time
    print(f"Speedup:  DP is {speedup:.2f}x faster than GA")
    
    # Complexity Analysis
    print("\n3. COMPLEXITY ANALYSIS")
    print("-" * 80)
    print(f"Problem Size: n={N_ITEMS} items, W={CAPACITY} capacity")
    print(f"\nDynamic Programming:")
    print(f"  Time Complexity:  O(n × W) = O({N_ITEMS} × {CAPACITY}) = {N_ITEMS * CAPACITY:,} operations")
    print(f"  Space Complexity: O(n × W) = {(N_ITEMS + 1) * (CAPACITY + 1):,} table entries")
    print(f"  Classification:   Pseudo-polynomial (polynomial in W's numeric value)")
    
    ga_ops = GENERATIONS * POPULATION_SIZE * N_ITEMS
    print(f"\nGenetic Algorithm:")
    print(f"  Time Complexity:  O(G × P × n) = O({GENERATIONS} × {POPULATION_SIZE} × {N_ITEMS})")
    print(f"                    ≈ {ga_ops:,} operations")
    print(f"  Space Complexity: O(P × n) = O({POPULATION_SIZE} × {N_ITEMS}) = {POPULATION_SIZE * N_ITEMS:,} bits")
    print(f"  Classification:   Heuristic approximation algorithm")
    
    # Memory Usage
    print("\n4. MEMORY USAGE ESTIMATE")
    print("-" * 80)
    dp_memory = (N_ITEMS + 1) * (CAPACITY + 1) * 8  # 8 bytes per integer
    ga_memory = POPULATION_SIZE * N_ITEMS  # bits
    print(f"DP Memory:  ~{dp_memory / 1024:.2f} KB (storing full table)")
    print(f"GA Memory:  ~{ga_memory / 8 / 1024:.2f} KB (storing population)")
    
    # Scalability
    print("\n5. SCALABILITY PREDICTIONS")
    print("-" * 80)
    
    # Predict for larger problems
    test_sizes = [(100, 500), (1000, 5000), (10000, 100000)]
    print(f"{'n items':<12} {'W capacity':<12} {'DP (n×W)':<15} {'GA (G×P×n)':<15} {'Recommendation':<20}")
    print("-" * 80)
    for n, w in test_sizes:
        dp_ops = n * w
        ga_ops_pred = GENERATIONS * POPULATION_SIZE * n
        if dp_ops < ga_ops_pred:
            rec = "Use DP"
        elif dp_ops < 10**7:
            rec = "Use DP"
        else:
            rec = "Use GA"
        print(f"{n:<12} {w:<12} {dp_ops:>14,} {ga_ops_pred:>14,} {rec:<20}")
    
    # Solution Characteristics
    print("\n6. SOLUTION CHARACTERISTICS")
    print("-" * 80)
    ga_selected = [i for i in range(N_ITEMS) if ga_solution[i] == 1]
    ga_weight = sum(WEIGHTS[i] for i in ga_selected)
    dp_weight = sum(WEIGHTS[i] for i in dp_items)
    
    print(f"DP Solution:")
    print(f"  Items Selected: {len(dp_items)}/{N_ITEMS}")
    print(f"  Total Weight:   {dp_weight}/{CAPACITY} ({dp_weight/CAPACITY*100:.1f}% capacity)")
    print(f"  Items:          {dp_items}")
    
    print(f"\nGA Solution:")
    print(f"  Items Selected: {len(ga_selected)}/{N_ITEMS}")
    print(f"  Total Weight:   {ga_weight}/{CAPACITY} ({ga_weight/CAPACITY*100:.1f}% capacity)")
    print(f"  Items:          {ga_selected}")
    
    # Recommendations
    print("\n7. RECOMMENDATIONS")
    print("-" * 80)
    print(f"For this problem (n={N_ITEMS}, W={CAPACITY}):")
    print(f"  → Use DYNAMIC PROGRAMMING")
    print(f"    Reasons:")
    print(f"    • Guarantees optimal solution")
    print(f"    • Faster execution ({speedup:.1f}x)")
    print(f"    • Small problem size (n×W = {N_ITEMS * CAPACITY:,})")
    
    print(f"\nGeneral Guidelines:")
    print(f"  • n × W < 10,000,000        → Always use DP")
    print(f"  • n × W > 100,000,000       → Always use GA")
    print(f"  • 10M < n × W < 100M        → Use GA first, validate with DP if needed")
    print(f"  • Optimal solution required → Always use DP (if feasible)")
    print(f"  • Time-constrained          → Use GA with tuned parameters")


def print_binary_comparison(ga_solution: List[int], dp_items: List[int]) -> None:
    """Print side-by-side binary representation of both solutions."""
    print("\n8. BINARY REPRESENTATION COMPARISON")
    print("-" * 80)
    
    # Create binary strings
    ga_binary = ''.join(map(str, ga_solution))
    dp_binary = ['0'] * N_ITEMS
    for idx in dp_items:
        dp_binary[idx] = '1'
    dp_binary = ''.join(dp_binary)
    
    print("Item Index: ", ' '.join(f"{i:2d}" for i in range(N_ITEMS)))
    print("DP Solution:", '  '.join(dp_binary))
    print("GA Solution:", '  '.join(ga_binary))
    
    # Highlight differences
    differences = sum(1 for i in range(N_ITEMS) if ga_binary[i] != dp_binary[i])
    print(f"\nDifferences: {differences}/{N_ITEMS} bits differ")
    
    if differences > 0:
        diff_indices = [i for i in range(N_ITEMS) if ga_binary[i] != dp_binary[i]]
        print(f"Different at indices: {diff_indices}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("0/1 KNAPSACK PROBLEM SOLVER")
    print("Comprehensive Comparison: Genetic Algorithm vs Dynamic Programming")
    print("=" * 80)
    
    # Display problem data
    print("\nPROBLEM SPECIFICATION")
    print("-" * 80)
    print(f"Number of Items:     {N_ITEMS}")
    print(f"Knapsack Capacity:   {CAPACITY}")
    print(f"Total Available Value: {sum(VALUES)}")
    print(f"Total Weight if all items: {sum(WEIGHTS)}")
    print(f"\nItem Details:")
    print(f"{'Item':<6} {'Weight':<8} {'Value':<8} {'Value/Weight':<12}")
    print("-" * 80)
    for i in range(N_ITEMS):
        ratio = VALUES[i] / WEIGHTS[i]
        print(f"{i:<6} {WEIGHTS[i]:<8} {VALUES[i]:<8} {ratio:.3f}")
    
    # Run Dynamic Programming
    print("\n" + "=" * 80)
    print("RUNNING DYNAMIC PROGRAMMING")
    print("=" * 80)
    dp_value, dp_weight, dp_items, dp_time = knapsack_dp()
    print(f"✓ Optimal solution found in {dp_time*1000:.4f} ms")
    print(f"  Maximum Value: {dp_value}")
    print(f"  Total Weight:  {dp_weight}/{CAPACITY}")
    print(f"  Items:         {dp_items}")
    
    # Run Genetic Algorithm
    print("\n" + "=" * 80)
    print("RUNNING GENETIC ALGORITHM")
    print("=" * 80)
    print(f"Parameters: Pop={POPULATION_SIZE}, Gen={GENERATIONS}, " +
          f"Mutation={MUTATION_RATE}, Crossover={CROSSOVER_RATE}")
    print()
    
    ga_solution, ga_value, ga_gen, ga_time, fitness_history = genetic_algorithm()
    ga_weight = get_weight(ga_solution)
    ga_items = [i for i in range(N_ITEMS) if ga_solution[i] == 1]
    
    print(f"\n✓ GA completed in {ga_time*1000:.4f} ms")
    print(f"  Best found at generation: {ga_gen}")
    print(f"  Final Value:  {ga_value}")
    print(f"  Total Weight: {ga_weight}/{CAPACITY}")
    print(f"  Items:        {ga_items}")
    
    # Perform comparative analysis
    compare_solutions(ga_solution, ga_value, ga_time, 
                     dp_value, dp_items, dp_time)
    
    # Binary comparison
    print_binary_comparison(ga_solution, dp_items)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ DP found optimal solution: Value = {dp_value}, Time = {dp_time*1000:.4f} ms")
    print(f"✓ GA found {'optimal' if ga_value == dp_value else 'suboptimal'} solution: " +
          f"Value = {ga_value}, Time = {ga_time*1000:.4f} ms")
    
    if ga_value == dp_value:
        print("\n🎯 Success! GA matched the optimal DP solution!")
    else:
        gap = ((dp_value - ga_value) / dp_value) * 100
        print(f"\n⚠️  GA solution is {gap:.2f}% below optimal")
        print(f"   Consider: Increasing population size, generations, or adjusting operators")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()