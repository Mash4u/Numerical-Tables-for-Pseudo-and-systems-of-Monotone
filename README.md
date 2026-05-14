# Numerical Tables for RSANME: Relaxed Self-Adaptive Newton Method with Error

This repository contains numerical experiments for the **RSANME (Relaxed Self-Adaptive Newton Method with Error)** algorithm for solving nonlinear equations and systems of monotone operators.

## Overview

RSANME is an iterative method that combines:
- Relaxation parameters for improved stability
- Self-adaptive step sizes for faster convergence
- Backtracking line search with Armijo conditions

## Features

- Complete implementation of the RSANME algorithm
- Multiple test problems from literature
- Comprehensive numerical experiments
- Convergence analysis and performance comparisons
- Visualization tools for results

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run all numerical experiments:

```bash
python run_experiments.py
```

This will:
1. Test RSANME with different parameter settings
2. Compare RSANME with classical Newton method
3. Generate numerical tables
4. Save results to `results.csv`

### Creating Visualizations

To create convergence plots and performance profiles:

```bash
python visualize.py
```

This generates:
- Convergence history plots for each test problem
- Performance profiles comparing different methods

### Using RSANME in Your Code

```python
from rsanme import RSANME
import numpy as np

# Define your function and Jacobian
def func(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

def jacobian(x):
    return np.array([[2*x[0], 2*x[1]], [1, -1]])

# Create solver
solver = RSANME(
    func=func,
    jacobian=jacobian,
    x0=np.array([0.5, 0.5]),
    tol=1e-6,
    max_iter=100,
    alpha=0.5,  # relaxation parameter
    beta=0.5,   # line search parameter
    gamma=0.9   # Armijo parameter
)

# Solve
x_solution, info = solver.solve()
print(f"Solution: {x_solution}")
print(f"Converged: {info['converged']}")
print(f"Iterations: {info['num_iterations']}")
```

## Test Problems

The repository includes several benchmark problems:

1. **Broyden System**: 3D nonlinear system
2. **Rosenbrock Gradient**: n-dimensional optimization problem
3. **Exponential System**: Simple exponential equations
4. **Tridiagonal System**: Sparse linear-like system
5. **Discrete Boundary Value Problem**: Discretized BVP

## Numerical Results

The experiments generate comprehensive tables showing:

- **Iteration counts**: Number of iterations to converge
- **Residual norms**: Final values of ||F(x)||
- **CPU time**: Computational efficiency
- **Success rates**: Percentage of problems solved
- **Convergence rates**: Speed of convergence

## Parameters

RSANME accepts the following parameters:

- `alpha` (default: 0.5): Relaxation parameter (0 < α < 1)
- `beta` (default: 0.5): Backtracking line search parameter
- `gamma` (default: 0.9): Armijo condition parameter
- `tol` (default: 1e-6): Convergence tolerance
- `max_iter` (default: 100): Maximum iterations

## Files

- `rsanme.py`: Core RSANME algorithm implementation
- `test_problems.py`: Test problem definitions
- `run_experiments.py`: Main experiment script
- `visualize.py`: Visualization tools
- `requirements.txt`: Python dependencies

## Contributing

Contributions are welcome! Feel free to:
- Add new test problems
- Improve the algorithm
- Enhance visualizations
- Fix bugs

## License

This project is open source and available for academic and research purposes.

## References

This implementation is based on research in numerical optimization and nonlinear equation solving, particularly work on:
- Newton-type methods
- Monotone operator theory
- Self-adaptive algorithms