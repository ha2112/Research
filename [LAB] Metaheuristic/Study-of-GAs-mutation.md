# Comprehensive Study of Genetic Operators

## Table of Contents

1. [Crossover Operators](#crossover-operators)
   - [Simulated Binary Crossover (SBX)](#simulated-binary-crossover-sbx)
   - [Cycle Crossover (CX)](#cycle-crossover-cx)
   - [Partially Mapped Crossover (PMX)](#partially-mapped-crossover-pmx)
   - [Order Crossover (OX)](#order-crossover-ox)
2. [Mutation Operators](#mutation-operators)
   - [Polynomial Mutation](#polynomial-mutation)

---

## Crossover Operators

### Simulated Binary Crossover (SBX)

#### Description

Simulated Binary Crossover (SBX) is a real-coded crossover operator that mimics the behavior of single-point crossover in binary-coded genetic algorithms. It was designed specifically for continuous optimization problems and creates offspring that maintain the statistical properties of the parent distribution. The operator ensures that offspring are more likely to be near their parents, with the spread controlled by a distribution index parameter.

#### Mechanism

**Step-by-step process:**

1. **Select two parents:** Choose parent chromosomes P₁ and P₂ with real-valued genes
2. **Generate random number:** For each gene position i, generate a random number u ∈ [0, 1]
3. **Calculate spread factor β:** Compute β using the distribution index ηc:
   - If u ≤ 0.5: β = (2u)^(1/(ηc+1))
   - If u > 0.5: β = (1/(2(1-u)))^(1/(ηc+1))
4. **Create offspring:** Generate two offspring C₁ and C₂:
   - C₁[i] = 0.5[(1 + β)P₁[i] + (1 - β)P₂[i]]
   - C₂[i] = 0.5[(1 - β)P₁[i] + (1 + β)P₂[i]]
5. **Apply bounds:** Ensure offspring values remain within variable bounds

**Example:**

```python
import numpy as np

def sbx_crossover(parent1, parent2, eta_c=20, bounds=(0, 1)):
    """
    Simulated Binary Crossover
    eta_c: distribution index (higher values = offspring closer to parents)
    """
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        u = np.random.random()

        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta_c + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))

        offspring1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        offspring2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

        # Apply bounds
        offspring1[i] = np.clip(offspring1[i], bounds[0], bounds[1])
        offspring2[i] = np.clip(offspring2[i], bounds[0], bounds[1])

    return offspring1, offspring2

# Example usage
P1 = np.array([0.3, 0.7, 0.5])
P2 = np.array([0.6, 0.4, 0.8])
C1, C2 = sbx_crossover(P1, P2, eta_c=20)

print(f"Parent 1: {P1}")
print(f"Parent 2: {P2}")
print(f"Child 1:  {C1}")
print(f"Child 2:  {C2}")
```

**Sample Output:**

```
Parent 1: [0.3 0.7 0.5]
Parent 2: [0.6 0.4 0.8]
Child 1:  [0.42 0.58 0.63]
Child 2:  [0.48 0.52 0.67]
```

#### Use Case

**Best suited for:**

- Real-valued optimization problems (continuous domains)
- Multi-objective optimization (widely used in NSGA-II, NSGA-III)
- Function optimization where variables are continuous
- Engineering design optimization
- Parameter tuning problems

**Chromosome representation:** Real-valued vectors (e.g., [0.5, 2.3, -1.7, 4.2])

#### Pros & Cons

**Advantages:**

- Self-adaptive spread control through distribution index
- Maintains solution diversity effectively
- Offspring tend to be feasible if parents are feasible
- Well-suited for multi-objective optimization
- Creates offspring near parents with controllable variance
- Theoretically sound with proven convergence properties

**Disadvantages:**

- Requires careful tuning of distribution index (ηc)
- Can lead to premature convergence if ηc is too high
- May struggle with highly constrained problems
- Computationally more expensive than simple arithmetic crossover
- Performance depends on proper parameter settings

---

### Cycle Crossover (CX)

#### Description

Cycle Crossover (CX) is a specialized crossover operator designed for permutation-based representations. It preserves the absolute position of elements from both parents by identifying cycles in the parent chromosomes. Each gene comes from one parent or the other, and the operator ensures that no position receives a duplicate value, making it ideal for problems where the position of elements matters.

#### Mechanism

**Step-by-step process:**

1. **Initialize:** Create two offspring arrays, initially empty
2. **Identify first cycle:**
   - Start at position 0
   - Copy gene from Parent 1 to Offspring 1 at this position
   - Find the position of this gene value in Parent 2
   - Move to that position and repeat until cycle completes
3. **Copy cycle:** For all positions in the cycle, copy from Parent 1 to Offspring 1
4. **Fill remaining:** For all positions NOT in the cycle, copy from Parent 2 to Offspring 1
5. **Create second offspring:** Swap parent roles (Offspring 2 gets cycle from Parent 2, rest from Parent 1)
6. **Repeat if needed:** If multiple cycles exist, alternate which parent contributes each cycle

**Example:**

```python
def cycle_crossover(parent1, parent2):
    """
    Cycle Crossover for permutation chromosomes
    """
    size = len(parent1)
    offspring1 = [-1] * size
    offspring2 = [-1] * size

    # Track which positions have been assigned
    visited = [False] * size

    # Identify cycles
    cycle_num = 0
    for start in range(size):
        if visited[start]:
            continue

        # Trace a cycle
        cycle = []
        pos = start
        while not visited[pos]:
            visited[pos] = True
            cycle.append(pos)
            # Find where parent1[pos] appears in parent2
            value = parent1[pos]
            pos = parent2.index(value)

        # Assign cycle alternately to offspring
        for idx in cycle:
            if cycle_num % 2 == 0:
                offspring1[idx] = parent1[idx]
                offspring2[idx] = parent2[idx]
            else:
                offspring1[idx] = parent2[idx]
                offspring2[idx] = parent1[idx]

        cycle_num += 1

    return offspring1, offspring2

# Example usage
P1 = [1, 2, 3, 4, 5, 6, 7, 8]
P2 = [3, 7, 5, 1, 6, 8, 2, 4]

print(f"Parent 1: {P1}")
print(f"Parent 2: {P2}")

C1, C2 = cycle_crossover(P1, P2)

print(f"Child 1:  {C1}")
print(f"Child 2:  {C2}")
```

**Detailed Example Trace:**

```
Parent 1: [1, 2, 3, 4, 5, 6, 7, 8]
Parent 2: [3, 7, 5, 1, 6, 8, 2, 4]

Cycle 1:
- Start at position 0: P1[0]=1
- Find 1 in P2: position 3
- P1[3]=4, find 4 in P2: position 7
- P1[7]=8, find 8 in P2: position 5
- P1[5]=6, find 6 in P2: position 4
- P1[4]=5, find 5 in P2: position 2
- P1[2]=3, find 3 in P2: position 0 (cycle complete!)

Cycle positions: {0, 3, 7, 5, 4, 2}

Child 1: [1, 7, 3, 4, 5, 6, 2, 8]  (cycle from P1, rest from P2)
Child 2: [3, 2, 5, 1, 6, 8, 7, 4]  (cycle from P2, rest from P1)
```

#### Use Case

**Best suited for:**

- Traveling Salesman Problem (TSP)
- Vehicle routing problems
- Scheduling problems where absolute positions matter
- Assignment problems
- Job shop scheduling

**Chromosome representation:** Permutations where each element appears exactly once and position is significant (e.g., [3, 1, 4, 2, 5])

#### Pros & Cons

**Advantages:**

- Preserves absolute positions from parents
- Always produces valid permutations (no duplicates)
- Maintains more genetic material from parents than many other operators
- Deterministic behavior (no randomness)
- Works well when position information is critical
- Simple to implement and understand

**Disadvantages:**

- Can be disruptive, potentially breaking good building blocks
- May not preserve relative order of elements
- Less effective when relative ordering matters more than absolute position
- Can lead to slower convergence in some problems
- Limited exploration capability compared to other permutation operators
- Not suitable for problems where adjacency relationships are important

---

### Partially Mapped Crossover (PMX)

#### Description

Partially Mapped Crossover (PMX) is a permutation crossover operator that preserves relative ordering and position information from both parents. It works by selecting a random segment from one parent and attempting to preserve the position of as many elements as possible while avoiding duplicates. PMX uses a mapping relationship to resolve conflicts, ensuring valid permutations are always produced.

#### Mechanism

**Step-by-step process:**

1. **Select two random cut points:** Choose two positions to define a crossover segment
2. **Copy segment:** Copy the segment between cut points from Parent 1 to Offspring 1
3. **Create mapping:** Establish a mapping relationship between elements in the segment:
   - Map P1[i] ↔ P2[i] for all positions i in the segment
4. **Fill remaining positions:** For each position outside the segment:
   - Try to copy the element from Parent 2
   - If it creates a duplicate, follow the mapping chain until a non-duplicate is found
5. **Create second offspring:** Repeat with parent roles reversed

**Example:**

```python
def pmx_crossover(parent1, parent2):
    """
    Partially Mapped Crossover for permutation chromosomes
    """
    size = len(parent1)
    offspring1 = [-1] * size
    offspring2 = [-1] * size

    # Select two random cut points
    cut1, cut2 = sorted(np.random.choice(range(size), 2, replace=False))

    # Copy segment from parent1 to offspring1
    offspring1[cut1:cut2] = parent1[cut1:cut2]
    offspring2[cut1:cut2] = parent2[cut1:cut2]

    # Create mapping
    def fill_offspring(offspring, p1, p2):
        mapping = {}
        for i in range(cut1, cut2):
            mapping[p1[i]] = p2[i]

        for i in range(size):
            if i >= cut1 and i < cut2:
                continue  # Already filled

            value = p2[i]
            # Follow mapping chain if value already exists
            while value in offspring[cut1:cut2]:
                value = mapping[value]

            offspring[i] = value

    fill_offspring(offspring1, parent1, parent2)
    fill_offspring(offspring2, parent2, parent1)

    return offspring1, offspring2

# Example usage
P1 = [1, 2, 3, 4, 5, 6, 7, 8]
P2 = [3, 7, 5, 1, 6, 8, 2, 4]

print(f"Parent 1: {P1}")
print(f"Parent 2: {P2}")

np.random.seed(42)
C1, C2 = pmx_crossover(P1, P2)

print(f"Child 1:  {C1}")
print(f"Child 2:  {C2}")
```

**Detailed Example Trace:**

```
Parent 1: [1, 2, 3, 4, 5, 6, 7, 8]
Parent 2: [3, 7, 5, 1, 6, 8, 2, 4]

Cut points: 3 and 6

Step 1: Copy segment
Child 1:  [-, -, -, 4, 5, 6, -, -]
Mapping: 4↔1, 5↔6, 6↔8

Step 2: Fill remaining positions from Parent 2
Position 0: Try 3 → not in segment → Child 1[0] = 3
Position 1: Try 7 → not in segment → Child 1[1] = 7
Position 2: Try 5 → in segment! Follow mapping: 5↔6, 6↔8, 8 not in segment → Child 1[2] = 8
Position 6: Try 2 → not in segment → Child 1[6] = 2
Position 7: Try 4 → in segment! Follow mapping: 4↔1, 1 not in segment → Child 1[7] = 1

Result:
Child 1: [3, 7, 8, 4, 5, 6, 2, 1]
Child 2: [4, 2, 6, 1, 5, 8, 7, 3]
```

#### Use Case

**Best suited for:**

- Traveling Salesman Problem (TSP)
- Vehicle routing problems
- Job shop scheduling
- Sequencing problems
- Timetabling applications
- Problems where relative ordering is important

**Chromosome representation:** Permutations where each element appears exactly once (e.g., [2, 4, 1, 3, 5])

#### Pros & Cons

**Advantages:**

- Preserves relative ordering better than Cycle Crossover
- Always produces valid permutations
- Good balance between exploitation and exploration
- Maintains position information from both parents
- Well-studied and proven effective for TSP
- Intuitive mapping mechanism

**Disadvantages:**

- More complex to implement than simpler crossover operators
- Can be computationally expensive for large chromosomes
- Mapping chain can sometimes be long and convoluted
- May not preserve adjacency information well
- Performance depends on cut point selection
- Can be disruptive to good subsequences spanning cut points

---

### Order Crossover (OX)

#### Description

Order Crossover (OX) is a permutation crossover operator that emphasizes preserving the relative order of elements from the parents. It selects a subsequence from one parent and fills the remaining positions with elements from the other parent in the order they appear. This makes it particularly effective for problems where the sequence or ordering of elements is more important than their absolute positions.

#### Mechanism

**Step-by-step process:**

1. **Select two random cut points:** Choose positions to define a crossover segment
2. **Copy segment:** Copy the segment between cut points from Parent 1 to Offspring 1
3. **Create ordered list:** Starting after the second cut point, list elements from Parent 2 in circular order
4. **Remove duplicates:** Remove any elements that are already in the copied segment
5. **Fill remaining:** Fill the remaining positions in Offspring 1 with the filtered list, starting after the second cut point (circular)
6. **Create second offspring:** Repeat with parent roles reversed

**Example:**

```python
def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) for permutation chromosomes
    """
    size = len(parent1)
    offspring1 = [-1] * size
    offspring2 = [-1] * size

    # Select two random cut points
    cut1, cut2 = sorted(np.random.choice(range(size), 2, replace=False))

    def create_offspring(offspring, p1, p2):
        # Copy segment from p1
        offspring[cut1:cut2] = p1[cut1:cut2]

        # Create list of elements from p2 starting after cut2
        p2_elements = list(p2[cut2:]) + list(p2[:cut2])

        # Remove elements already in offspring
        remaining = [e for e in p2_elements if e not in offspring[cut1:cut2]]

        # Fill remaining positions
        current_pos = cut2
        for element in remaining:
            offspring[current_pos % size] = element
            current_pos += 1

    create_offspring(offspring1, parent1, parent2)
    create_offspring(offspring2, parent2, parent1)

    return offspring1, offspring2

# Example usage
P1 = [1, 2, 3, 4, 5, 6, 7, 8]
P2 = [3, 7, 5, 1, 6, 8, 2, 4]

print(f"Parent 1: {P1}")
print(f"Parent 2: {P2}")

np.random.seed(42)
C1, C2 = order_crossover(P1, P2)

print(f"Child 1:  {C1}")
print(f"Child 2:  {C2}")
```

**Detailed Example Trace:**

```
Parent 1: [1, 2, 3, 4, 5, 6, 7, 8]
Parent 2: [3, 7, 5, 1, 6, 8, 2, 4]

Cut points: 3 and 6

Step 1: Copy segment from Parent 1
Child 1:  [-, -, -, 4, 5, 6, -, -]

Step 2: Create ordered list from Parent 2 starting after position 6
P2 from position 6: [2, 4] + [3, 7, 5, 1, 6, 8] = [2, 4, 3, 7, 5, 1, 6, 8]

Step 3: Remove elements already in Child 1 (4, 5, 6)
Remaining: [2, 3, 7, 1, 8]

Step 4: Fill starting from position 6 (circular)
Position 6: 2
Position 7: 3
Position 0: 7
Position 1: 1
Position 2: 8

Result:
Child 1: [7, 1, 8, 4, 5, 6, 2, 3]
Child 2: [5, 6, 7, 1, 2, 3, 8, 4]
```

#### Use Case

**Best suited for:**

- Traveling Salesman Problem (TSP) - particularly effective
- Sequential ordering problems
- Routing and scheduling where order matters
- Assembly line sequencing
- Task scheduling problems
- Path planning problems

**Chromosome representation:** Permutations where relative order is crucial (e.g., [3, 1, 4, 2, 5])

#### Pros & Cons

**Advantages:**

- Excellent at preserving relative ordering from parents
- Less disruptive than PMX to good subsequences
- Always produces valid permutations
- Simple and intuitive mechanism
- Well-suited for problems with strong ordering dependencies
- Computationally efficient
- Widely used and proven effective for TSP

**Disadvantages:**

- Does not preserve absolute position information well
- May not preserve adjacency relationships between non-consecutive elements
- Can be slow to converge on some problem types
- Limited preservation of building blocks that span cut points
- Performance can be sensitive to cut point selection
- May require more generations to reach optimal solutions than PMX in some cases

---

## Mutation Operators

### Polynomial Mutation

#### Description

Polynomial Mutation is a real-coded mutation operator designed for continuous optimization problems. It simulates the effect of bit-flip mutation in binary strings for real-valued variables. The operator adds a small perturbation to gene values using a polynomial probability distribution, with the mutation step size controlled by a distribution index parameter. This ensures that small mutations are more likely than large ones, promoting fine-tuning of solutions.

#### Mechanism

**Step-by-step process:**

1. **For each gene:** Iterate through all genes in the chromosome
2. **Check mutation probability:** Generate random number r ∈ [0, 1]. If r < pm (mutation probability), mutate this gene
3. **Generate random value:** Generate u ∈ [0, 1]
4. **Calculate delta:** Compute the perturbation value δ using distribution index ηm:
   - If u < 0.5: δ = (2u)^(1/(ηm+1)) - 1
   - If u ≥ 0.5: δ = 1 - (2(1-u))^(1/(ηm+1))
5. **Apply mutation:** Calculate mutated value:
   - x'[i] = x[i] + δ × (upper_bound - lower_bound)
6. **Apply bounds:** Ensure the mutated value stays within variable bounds

**Example:**

```python
import numpy as np

def polynomial_mutation(individual, eta_m=20, prob_m=0.1, bounds=(0, 1)):
    """
    Polynomial Mutation for real-valued chromosomes

    Parameters:
    - individual: array of real values to mutate
    - eta_m: distribution index (higher = smaller mutations)
    - prob_m: probability of mutating each gene
    - bounds: tuple of (lower_bound, upper_bound)
    """
    mutated = individual.copy()
    lower, upper = bounds
    delta = upper - lower

    for i in range(len(mutated)):
        if np.random.random() < prob_m:
            u = np.random.random()

            if u < 0.5:
                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))

            mutated[i] = mutated[i] + delta_q * delta

            # Apply bounds
            mutated[i] = np.clip(mutated[i], lower, upper)

    return mutated

# Example usage
individual = np.array([0.3, 0.7, 0.5, 0.9, 0.2])

print(f"Original:  {individual}")
print(f"\nMutations with different eta_m values:")

np.random.seed(42)
mutated_20 = polynomial_mutation(individual, eta_m=20, prob_m=1.0)
print(f"eta_m=20:  {mutated_20}")

np.random.seed(42)
mutated_100 = polynomial_mutation(individual, eta_m=100, prob_m=1.0)
print(f"eta_m=100: {mutated_100}")

np.random.seed(42)
mutated_5 = polynomial_mutation(individual, eta_m=5, prob_m=1.0)
print(f"eta_m=5:   {mutated_5}")

# Example with typical mutation probability
np.random.seed(10)
mutated_typical = polynomial_mutation(individual, eta_m=20, prob_m=0.2)
print(f"\nWith prob_m=0.2: {mutated_typical}")
```

**Sample Output:**

```
Original:  [0.3 0.7 0.5 0.9 0.2]

Mutations with different eta_m values:
eta_m=20:  [0.283 0.742 0.521 0.881 0.267]  (small changes)
eta_m=100: [0.296 0.708 0.504 0.895 0.211]  (very small changes)
eta_m=5:   [0.241 0.821 0.573 0.835 0.412]  (larger changes)

With prob_m=0.2: [0.3 0.7 0.521 0.9 0.2]  (only some genes mutated)
```

**Visualization of Distribution:**

```python
import matplotlib.pyplot as plt

def plot_mutation_distribution():
    """Visualize how eta_m affects mutation distribution"""
    u_values = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, eta_m in enumerate([5, 20, 100]):
        delta_q = np.zeros_like(u_values)
        for i, u in enumerate(u_values):
            if u < 0.5:
                delta_q[i] = (2 * u) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta_q[i] = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))

        axes[idx].plot(u_values, delta_q)
        axes[idx].set_title(f'eta_m = {eta_m}')
        axes[idx].set_xlabel('Random value (u)')
        axes[idx].set_ylabel('Perturbation (delta_q)')
        axes[idx].grid(True)
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
```

#### Use Case

**Best suited for:**

- Real-valued optimization problems
- Multi-objective optimization (complements SBX in NSGA-II, NSGA-III)
- Function optimization with continuous parameters
- Fine-tuning of near-optimal solutions
- Engineering design optimization
- Parameter optimization in machine learning

**Chromosome representation:** Real-valued vectors (e.g., [2.5, -1.3, 0.8, 4.2])

**Typical parameter settings:**

- Distribution index (ηm): 20-100 for exploitation, 5-10 for exploration
- Mutation probability (pm): 1/n where n is the number of variables

#### Pros & Cons

**Advantages:**

- Self-adaptive mutation strength through distribution index
- Smaller mutations are more probable (good for fine-tuning)
- Maintains diversity in the population
- Theoretically sound with polynomial probability distribution
- Complements SBX crossover well in real-coded GAs
- Controllable exploration vs. exploitation trade-off
- Can escape local optima through larger (but rare) mutations

**Disadvantages:**

- Requires careful tuning of distribution index (ηm)
- May converge slowly if ηm is too high
- Can be disruptive if ηm is too low
- Mutation probability needs problem-specific adjustment
- Not suitable for discrete or permutation problems
- May produce infeasible solutions if bounds are tight
- Performance sensitive to parameter settings

---

## Summary Comparison Table

| Operator       | Type      | Representation | Best For                       | Key Strength              | Main Limitation                |
| -------------- | --------- | -------------- | ------------------------------ | ------------------------- | ------------------------------ |
| **SBX**        | Crossover | Real-valued    | Continuous optimization        | Self-adaptive spread      | Parameter tuning required      |
| **CX**         | Crossover | Permutation    | TSP, absolute position matters | Preserves positions       | Disruptive to building blocks  |
| **PMX**        | Crossover | Permutation    | TSP, routing                   | Balances order & position | Complex implementation         |
| **OX**         | Crossover | Permutation    | Sequential problems            | Preserves relative order  | Loses position info            |
| **Polynomial** | Mutation  | Real-valued    | Continuous optimization        | Fine-tuning capability    | Needs careful parameter tuning |

---

## Practical Recommendations

### For Continuous Optimization:

- Use **SBX** for crossover with **Polynomial Mutation**
- Set ηc = 15-30 for crossover, ηm = 20 for mutation
- This combination is the foundation of NSGA-II and NSGA-III

### For Traveling Salesman Problem:

- **First choice:** Order Crossover (OX) - best for preserving tours
- **Alternative:** PMX if position information is critical
- Avoid CX unless absolute positions are specifically important

### For Scheduling/Sequencing:

- **OX** for problems where task order matters most
- **PMX** for problems where time slots (positions) matter
- Combine with swap or insertion mutation

### General Tips:

- Always validate that offspring are feasible after applying operators
- Use elitism to preserve best solutions
- Balance exploitation (high η values) and exploration (low η values)
- Consider problem-specific operators for specialized domains
