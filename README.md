# Genetic Algorithm for Job Shop Scheduling

This project implements a Genetic Algorithm (GA) to solve the Job Shop Scheduling Problem (JSSP). The implementation is based on benchmark datasets from Kacem and Brandimarte, and includes comprehensive error handling and validation.

## Overview

The Job Shop Scheduling Problem is a classic optimization problem in operations research. It involves scheduling a set of jobs on a set of machines with the objective of minimizing the makespan (total completion time). Each job consists of a sequence of operations that must be processed on specific machines in a given order.

## Requirements

- Python 3.7+
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the benchmark datasets in the following structure:
```
project GA/
├── brandimarte/
│   ├── mk01.txt
│   ├── mk02.txt
│   └── ...
├── kacem/
│   ├── kacem01.txt
│   └── ...
└── brandimarte_kacem.json
```

## Algorithm Components

### 1. Representation
- Each solution (chromosome) is represented as a permutation of job indices
- The length of the chromosome equals the number of jobs
- Each gene represents the position of a job in the schedule

### 2. Fitness Function
- The fitness function calculates the makespan of a given schedule
- Makespan is computed by simulating the schedule on all machines
- Each machine maintains a timeline of operations
- Operations cannot overlap on the same machine
- The makespan is the maximum completion time across all machines

### 3. Genetic Operators

#### a. Selection
- Tournament selection is used for parent selection
- Two random individuals are chosen and the better one is selected
- This maintains selection pressure while preserving diversity

#### b. Crossover
- Order-based crossover (OX) is implemented
- Process:
  1. Randomly select two crossover points
  2. Copy the segment between points from parent 1 to child
  3. Fill remaining positions with genes from parent 2 in order
  4. Ensure no duplicate jobs in the child
- This preserves subsequences from parent solutions

#### c. Mutation
- Swap mutation is implemented
- Randomly selects two positions and swaps their values
- Mutation rate is configurable (default: 0.1)
- Helps maintain population diversity

### 4. Population Management
- Population size is configurable (default: 100)
- Each generation consists of:
  1. Selection of parents
  2. Crossover to create offspring
  3. Mutation of offspring
  4. Replacement of old population
- Best solution is tracked across generations

## Usage

Run the genetic algorithm:
```bash
python genetic_algorithm.py
```

The program will automatically process all instances defined in `brandimarte_kacem.json` and display the results for each instance.

## Features

1. **Genetic Algorithm Implementation**
   - Order-based crossover
   - Swap mutation
   - Tournament selection
   - Elitism (preservation of best solutions)
   - Configurable parameters:
     - Population size
     - Mutation rate
     - Crossover rate


2. **Input Handling**
   - Supports both Kacem and Brandimarte benchmark datasets
   - Automatic detection of input format
   - Comprehensive error handling
   - Input validation

3. **Output and Analysis**
   - Best makespan found for each instance
   - Comparison with optimal solutions (when available)
   - Generation-by-generation progress tracking
   - Solution visualization

## Results and Performance Metrics

The program provides comprehensive results and performance analysis for each instance. Here's a breakdown of the key metrics:

### 1. Solution Quality Metrics
- **Optimal Makespan**: If available, the known optimal solution for the instance
- **Optimality Gap**: Percentage difference between the solution found and the optimal makespan
- **Upper and Lower Bounds**: For instances where exact optimum is unknown, the program provides:
  - Upper bound: Best known solution
  - Lower bound: Minimum possible makespan
  - Gap calculations from both bounds

### 2. Solution Performance
- **Best Makespan Found**: The minimum makespan achieved by the genetic algorithm
- **Initial Fitness**: Starting fitness value before optimization
- **Improvement**: Absolute improvement in makespan from initial to final solution
- **Improvement Percentage**: Relative improvement as a percentage of initial makespan

### 3. Performance Analysis
- **Generation Statistics**: Detailed evolution of solution quality across generations
- **Convergence Rate**: How quickly the algorithm converges to good solutions
- **Solution Stability**: Consistency of results across multiple runs
- **Runtime Analysis**: Execution time and computational efficiency

### 4. Benchmark Comparison
- Comparison with known optimal solutions (when available)
- Performance relative to benchmark bounds
- Solution quality metrics across different instance sizes
- Comparison of different algorithm configurations

## Performance Metrics

The program tracks several key performance indicators:

1. **Solution Quality Metrics**
   - Makespan: Total completion time of all jobs
   - Optimality Gap: Percentage difference from optimal solution
   - Bound Gaps: Differences from upper and lower bounds

2. **Algorithm Performance**
   - Convergence Speed: How quickly good solutions are found
   - Solution Quality: How close to optimal solutions are found
   - Robustness: Consistency of results across multiple runs
   - Scalability: Performance with different problem sizes

3. **Computational Metrics**
   - Execution Time: Total runtime of the algorithm
   - Generation Time: Time taken per generation
   - Memory Usage: Resource consumption during execution

4. **Solution Characteristics**
   - Makespan Distribution: Range of solutions found
   - Generation Evolution: How solutions improve over time
   - Population Diversity: Genetic diversity maintained during evolution

## Result Interpretation

When interpreting results:
1. **Optimality Gap**
   - A gap of 0% indicates an optimal solution was found
   - Smaller gaps indicate better solution quality
   - Gaps from bounds provide confidence intervals for solution quality

2. **Improvement Metrics**
   - Higher improvement values indicate better optimization
   - Improvement percentage shows relative effectiveness
   - Comparison with initial fitness shows algorithm's effectiveness

3. **Bound Analysis**
   - Solutions closer to lower bounds are better
   - Gaps from upper bounds show how close to best known solutions
   - Both bounds provide a range of solution quality

4. **Generation Statistics**
   - Quick convergence indicates efficient algorithm
   - Stable improvements show consistent performance
   - Generation count shows computational effort

## Error Handling

The implementation includes:
- Input file validation
- Parameter validation
- Exception handling for file operations
- Runtime error detection
- Invalid solution detection

## Performance Considerations

- The algorithm uses NumPy for efficient array operations
- Memory usage is optimized by reusing arrays
- Computation is vectorized where possible
- Early stopping is implemented if optimal solution is found

## Future Improvements

Potential enhancements:
1. Parallel processing for multiple instances
2. Adaptive parameter tuning
3. Hybridization with local search
4. Visualization of solution evolution
5. Additional benchmark datasets
6. More sophisticated crossover operators
7. Advanced mutation strategies

## Contact

For any questions, suggestions, or issues related to this project, please feel free to contact:

- Email: [your.email@example.com](yassine.elalami5@usmba.ac.ma)
- GitHub: [your-github-username](https://github.com/YassineElALAMI)

We welcome contributions and feedback to improve this implementation of the Genetic Algorithm for Job Shop Scheduling.

## References

1. Kacem, I., Hammadi, S., & Borne, P. (2002). Approach by localization and multiobjective evolutionary optimization for flexible job-shop scheduling problems. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 32(1), 1-13.
2. Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search. Annals of Operations Research, 41(3), 157-183.
