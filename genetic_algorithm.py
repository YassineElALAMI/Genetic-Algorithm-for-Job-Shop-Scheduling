# Import necessary libraries
import numpy as np  # Used for numerical operations and random choice
import json         # To load instance configuration in JSON format
import os           # For file path operations (not used in this script)
from typing import List, Tuple, Dict  # For type hinting
import time         # For performance timing
import random       # For additional randomization methods

class JobShopGA:
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1,
                crossover_rate: float = 0.8, max_generations: int = 1000,
                elitism_rate: float = 0.1, tournament_size: int = 3):
        """
        Initialize the genetic algorithm with enhanced parameters.
        Added elitism and configurable tournament size for better selection.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.elitism_rate = elitism_rate  # Percentage of best individuals to preserve
        self.tournament_size = tournament_size  # Size of tournament selection
        self.population = []          # Current population (list of chromosomes)
        self.best_solution = None     # Best chromosome found
        self.best_fitness = float('inf')  # Fitness of the best chromosome (min makespan)
        self.fitness_history = []     # Track fitness evolution
        self.stagnation_counter = 0   # Track generations without improvement
        self.adaptive_mutation = mutation_rate  # Dynamic mutation rate

    def load_instance(self, instance_path: str) -> Tuple[int, int, List[List[List[int]]]]:
        """
        Load a job shop scheduling problem instance from a file.
        Enhanced with better error handling and validation.
        """
        try:
            with open(instance_path, 'r') as f:
                lines = f.readlines()

            # First line contains the number of jobs and machines
            n_jobs, n_machines = map(int, lines[0].split())
            operations = []

            # Parse each line to extract operations for jobs
            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split()
                job = []
                # Every two values represent one operation: machine and duration
                for j in range(0, len(parts), 2):
                    if j + 1 < len(parts):  # Avoid index out-of-range
                        machine = int(parts[j]) - 1  # Machine index is 0-based
                        duration = int(parts[j + 1])
                        if machine < 0:
                            raise ValueError(f"Invalid operation at line {i}: machine={machine+1}, duration={duration}")
                        job.append([machine, duration])
                operations.append(job)

            if len(operations) != n_jobs:
                raise ValueError(f"Expected {n_jobs} jobs, found {len(operations)}")

            return n_jobs, n_machines, operations
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Instance file not found: {instance_path}")
        except Exception as e:
            raise ValueError(f"Error parsing instance file: {e}")

    def initialize_population(self, n_jobs: int) -> List[List[int]]:
        """
        Generate diverse initial population using multiple strategies.
        Enhanced with different initialization methods for better diversity.
        """
        population = []
        
        # Strategy 1: Random permutations (70% of population)
        random_count = int(0.7 * self.population_size)
        for _ in range(random_count):
            chromosome = list(range(n_jobs))
            np.random.shuffle(chromosome)
            population.append(chromosome)
        
        # Strategy 2: Sorted by job indices (10% of population)
        sorted_count = int(0.1 * self.population_size)
        for _ in range(sorted_count):
            chromosome = list(range(n_jobs))
            population.append(chromosome)
        
        # Strategy 3: Reverse sorted (10% of population)
        reverse_count = int(0.1 * self.population_size)
        for _ in range(reverse_count):
            chromosome = list(range(n_jobs))[::-1]
            population.append(chromosome)
        
        # Strategy 4: Partially sorted with random swaps (10% of population)
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            chromosome = list(range(n_jobs))
            # Apply random swaps to create partial order
            for _ in range(n_jobs // 4):
                i, j = np.random.choice(n_jobs, 2, replace=False)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            population.append(chromosome)
        
        return population

    def calculate_fitness(self, chromosome: List[int], operations: List[List[List[int]]]) -> int:
        """
        Enhanced fitness calculation with optimized scheduling simulation.
        Added job completion time tracking for better scheduling accuracy.
        """
        n_jobs = len(operations)
        max_machine = max(max(op[0] for op in job) for job in operations)

        # Track machine availability and job completion times
        machine_times = np.zeros(max_machine, dtype=int)
        job_times = np.zeros(n_jobs, dtype=int)

        # Process operations in chromosome order
        for job_idx in chromosome:
            current_job_time = job_times[job_idx]
            
            for op_idx, (machine, duration) in enumerate(operations[job_idx]):
                # Convert to 0-based indexing for array access
                machine_idx = machine - 1
                # Operation can start when both job and machine are ready
                start_time = max(current_job_time, machine_times[machine_idx])
                completion_time = start_time + duration
                
                # Update times
                machine_times[machine_idx] = completion_time
                job_times[job_idx] = completion_time
                current_job_time = completion_time

        # Makespan is the maximum completion time across all machines
        return int(np.max(machine_times))

    def tournament_selection(self, fitnesses: List[int]) -> int:
        """
        Enhanced tournament selection with configurable tournament size.
        """
        candidates = np.random.choice(self.population_size, self.tournament_size, replace=False)
        best_idx = candidates[0]
        best_fitness = fitnesses[best_idx]
        
        for candidate in candidates[1:]:
            if fitnesses[candidate] < best_fitness:
                best_fitness = fitnesses[candidate]
                best_idx = candidate
        
        return best_idx

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Enhanced order crossover (OX) with improved segment selection.
        """
        n = len(parent1)
        if n <= 2:
            return parent1.copy(), parent2.copy()
        
        # Ensure reasonable segment size (20-80% of chromosome length)
        min_segment = max(1, n // 5)
        max_segment = min(n - 1, 4 * n // 5)
        segment_length = np.random.randint(min_segment, max_segment + 1)
        start = np.random.randint(0, n - segment_length + 1)
        end = start + segment_length

        child1 = [-1] * n
        child2 = [-1] * n

        # Copy segments from parents
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill remaining positions maintaining order
        self._fill_child(child1, parent2, end, n)
        self._fill_child(child2, parent1, end, n)

        return child1, child2

    def _fill_child(self, child: List[int], parent: List[int], start_pos: int, n: int):
        """Helper method to fill child chromosome maintaining parent order."""
        pos = start_pos % n
        for gene in parent:
            if gene not in child:
                while child[pos] != -1:
                    pos = (pos + 1) % n
                child[pos] = gene
                pos = (pos + 1) % n

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Enhanced mutation with multiple strategies and adaptive rates.
        """
        mutated = chromosome.copy()
        
        if np.random.rand() < self.adaptive_mutation:
            mutation_type = np.random.choice(['swap', 'insert', 'inversion'])
            
            if mutation_type == 'swap':
                # Original swap mutation
                idx1, idx2 = np.random.choice(len(chromosome), 2, replace=False)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
            
            elif mutation_type == 'insert':
                # Insert mutation: remove element and insert at different position
                if len(chromosome) > 2:
                    remove_idx = np.random.randint(len(chromosome))
                    element = mutated.pop(remove_idx)
                    insert_idx = np.random.randint(len(chromosome))
                    mutated.insert(insert_idx, element)
            
            elif mutation_type == 'inversion':
                # Inversion mutation: reverse a segment
                if len(chromosome) > 2:
                    start = np.random.randint(len(chromosome) - 1)
                    end = np.random.randint(start + 1, len(chromosome))
                    mutated[start:end+1] = mutated[start:end+1][::-1]
        
        return mutated

    def adaptive_parameters(self, generation: int):
        """
        Adapt mutation rate and other parameters based on progress.
        """
        # Increase mutation rate if stagnating
        if self.stagnation_counter > 50:
            self.adaptive_mutation = min(0.3, self.mutation_rate * 1.5)
        elif self.stagnation_counter > 20:
            self.adaptive_mutation = min(0.2, self.mutation_rate * 1.2)
        else:
            self.adaptive_mutation = self.mutation_rate

    def run(self, instance_path: str) -> Tuple[List[int], int]:
        """
        Enhanced main loop with elitism, adaptive parameters, and early stopping.
        """
        start_time = time.time()
        
        # Load instance data
        n_jobs, n_machines, operations = self.load_instance(instance_path)
        self.population = self.initialize_population(n_jobs)
        
        # Reset tracking variables
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.stagnation_counter = 0
        
        print(f"Starting GA with {n_jobs} jobs, {n_machines} machines")
        print(f"Population size: {self.population_size}, Generations: {self.max_generations}")

        for generation in range(self.max_generations):
            # Evaluate fitness of all chromosomes
            fitnesses = [self.calculate_fitness(chrom, operations) for chrom in self.population]
            
            # Track fitness evolution
            current_best = min(fitnesses)
            self.fitness_history.append(current_best)
            
            # Update best solution
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[fitnesses.index(current_best)].copy()
                self.stagnation_counter = 0
                print(f"Generation {generation}: New best makespan = {self.best_fitness}")
            else:
                self.stagnation_counter += 1

            # Adaptive parameter adjustment
            self.adaptive_parameters(generation)
            
            # Early stopping if no improvement for long time
            if self.stagnation_counter > 200:
                print(f"Early stopping at generation {generation} due to stagnation")
                break

            # Selection with elitism
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            
            # Get elite individuals
            elite_indices = np.argsort(fitnesses)[:elite_count]
            elite_population = [self.population[i].copy() for i in elite_indices]
            
            # Tournament selection for remaining population
            selected = []
            for _ in range(self.population_size - elite_count):
                selected_idx = self.tournament_selection(fitnesses)
                selected.append(self.population[selected_idx].copy())

            # Generate new population
            new_population = elite_population.copy()  # Keep elite
            
            # Crossover and mutation for non-elite
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, len(selected) - 1)]
                
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            
            # Ensure population size consistency
            self.population = new_population[:self.population_size]
            
            # Progress reporting
            if generation % 50 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"Generation {generation}: Best={self.best_fitness}, Avg={avg_fitness:.1f}, "
                      f"Mutation={self.adaptive_mutation:.3f}")

        elapsed_time = time.time() - start_time
        print(f"GA completed in {elapsed_time:.2f} seconds")
        print(f"Final best makespan: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness

    def get_statistics(self) -> Dict:
        """
        Return comprehensive statistics about the GA run.
        """
        if not self.fitness_history:
            return {}
        
        return {
            'best_fitness': self.best_fitness,
            'initial_fitness': self.fitness_history[0],
            'improvement': self.fitness_history[0] - self.best_fitness,
            'improvement_percentage': ((self.fitness_history[0] - self.best_fitness) / self.fitness_history[0]) * 100,
            'generations_run': len(self.fitness_history),
            'convergence_generation': self.fitness_history.index(self.best_fitness),
            'final_stagnation': self.stagnation_counter
        }

def main():
    """
    Enhanced main function with better reporting and error handling.
    """
    try:
        with open('brandimarte_kacem.json', 'r') as f:
            instances = json.load(f)
    except FileNotFoundError:
        print("Error: brandimarte_kacem.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in brandimarte_kacem.json")
        return

    # Enhanced GA parameters
    ga = JobShopGA(
        population_size=150,      # Increased population size
        mutation_rate=0.15,       # Slightly higher mutation rate
        crossover_rate=0.85,      # Higher crossover rate
        max_generations=1500,     # More generations
        elitism_rate=0.1,         # 10% elitism
        tournament_size=3         # Tournament size of 3
    )

    results = []
    total_start_time = time.time()

    for i, instance in enumerate(instances, 1):
        print(f"\n{'='*60}")
        print(f"Solving instance {i}/{len(instances)}: {instance['name']}")
        print(f"{'='*60}")
        
        try:
            solution, makespan = ga.run(instance['path'])
            stats = ga.get_statistics()
            
            # Calculate optimality gap if optimum is known
            gap = None
            if 'optimum' in instance and instance['optimum'] is not None:
                gap = ((makespan - instance['optimum']) / instance['optimum']) * 100
                print(f"Optimal makespan: {instance['optimum']}")
                print(f"Optimality gap: {gap:.2f}%")
            elif 'bounds' in instance:
                print(f"Upper bound: {instance['bounds']['upper']}")
                print(f"Lower bound: {instance['bounds']['lower']}")
                # Calculate gap from bounds if available
                if instance['bounds']['upper'] is not None:
                    upper_gap = ((makespan - instance['bounds']['lower']) / instance['bounds']['lower']) * 100
                    print(f"Gap from lower bound: {upper_gap:.2f}%")
                if instance['bounds']['upper'] is not None:
                    lower_gap = ((makespan - instance['bounds']['upper']) / instance['bounds']['upper']) * 100
                    print(f"Gap from upper bound: {lower_gap:.2f}%")
                gap = upper_gap if instance['bounds']['upper'] is not None else None
            
            # Display comprehensive results
            print(f"\nResults Summary:")
            print(f"Best makespan found: {makespan}")
            print(f"Initial fitness: {stats.get('initial_fitness', 'N/A')}")
            print(f"Improvement: {stats.get('improvement', 'N/A')}")
            print(f"Improvement %: {stats.get('improvement_percentage', 0):.2f}%")
            print(f"Convergence at generation: {stats.get('convergence_generation', 'N/A')}")
            
            results.append({
                'instance': instance['name'],
                'makespan': makespan,
                'optimum': instance.get('optimum'),
                'gap': gap,
                'stats': stats
            })
            
        except Exception as e:
            print(f"Error solving instance {instance['name']}: {e}")
            continue

    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Instances solved: {len(results)}/{len(instances)}")
    
    if results:
        gaps = []
        upper_gaps = []
        lower_gaps = []
        
        for result in results:
            if result['gap'] is not None:
                gaps.append(result['gap'])
            if result.get('upper_gap') is not None:
                upper_gaps.append(result['upper_gap'])
            if result.get('lower_gap') is not None:
                lower_gaps.append(result['lower_gap'])
        
        if gaps:
            print(f"Average optimality gap: {np.mean(gaps):.2f}%")
            print(f"Best optimality gap: {min(gaps):.2f}%")
            print(f"Worst optimality gap: {max(gaps):.2f}%")
            
        if upper_gaps:
            print(f"Average upper bound gap: {np.mean(upper_gaps):.2f}%")
            print(f"Best upper bound gap: {min(upper_gaps):.2f}%")
            print(f"Worst upper bound gap: {max(upper_gaps):.2f}%")
            
        if lower_gaps:
            print(f"Average lower bound gap: {np.mean(lower_gaps):.2f}%")
            print(f"Best lower bound gap: {min(lower_gaps):.2f}%")
            print(f"Worst lower bound gap: {max(lower_gaps):.2f}%")

    return

# Start the program
if __name__ == "__main__":
    main()