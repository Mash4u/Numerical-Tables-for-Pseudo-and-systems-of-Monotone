"""
Numerical experiments for RSANME algorithm.

This script runs comprehensive numerical experiments comparing RSANME
with classical Newton method on various test problems.
"""

import numpy as np
import pandas as pd
from rsanme import RSANME, modified_newton_method
from test_problems import get_test_problems
import time


def run_single_experiment(problem, method='rsanme', params=None):
    """
    Run a single experiment on a given problem.
    
    Parameters:
    -----------
    problem : TestProblem
        The test problem to solve
    method : str
        Method to use ('rsanme' or 'newton')
    params : dict
        Parameters for the method
        
    Returns:
    --------
    results : dict
        Dictionary containing results
    """
    if params is None:
        params = {}
    
    x0 = problem.initial_guess()
    
    # Start timing
    start_time = time.time()
    
    if method == 'rsanme':
        solver = RSANME(
            func=problem.func,
            jacobian=problem.jacobian,
            x0=x0,
            tol=params.get('tol', 1e-6),
            max_iter=params.get('max_iter', 100),
            alpha=params.get('alpha', 0.5),
            beta=params.get('beta', 0.5),
            gamma=params.get('gamma', 0.9)
        )
        x_sol, info = solver.solve()
    elif method == 'newton':
        x_sol, info = modified_newton_method(
            func=problem.func,
            jacobian=problem.jacobian,
            x0=x0,
            tol=params.get('tol', 1e-6),
            max_iter=params.get('max_iter', 100)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed_time = time.time() - start_time
    
    # Compute final residual
    final_residual = np.linalg.norm(problem.func(x_sol))
    
    # Compute error if exact solution is known
    exact_sol = problem.exact_solution()
    if exact_sol is not None:
        error = np.linalg.norm(x_sol - exact_sol)
    else:
        error = None
    
    results = {
        'problem': problem.name,
        'method': method,
        'converged': info.get('converged', False),
        'iterations': info.get('num_iterations', 0),
        'final_residual': final_residual,
        'error': error,
        'cpu_time': elapsed_time,
        'residual_history': info.get('residuals', [])
    }
    
    return results


def run_experiments():
    """
    Run all numerical experiments.
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame containing all experimental results
    """
    problems = get_test_problems()
    all_results = []
    
    methods = [
        ('rsanme', {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.9}),
        ('rsanme', {'alpha': 0.7, 'beta': 0.5, 'gamma': 0.9}),
        ('rsanme', {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.9}),
        ('newton', {})
    ]
    
    print("Running numerical experiments...")
    print("=" * 80)
    
    for problem in problems:
        print(f"\nProblem: {problem.name}")
        print("-" * 80)
        
        for method_name, params in methods:
            method_label = method_name
            if method_name == 'rsanme':
                method_label = f"{method_name} (α={params['alpha']})"
            
            try:
                results = run_single_experiment(problem, method_name, params)
                results['method_label'] = method_label
                all_results.append(results)
                
                status = "✓" if results['converged'] else "✗"
                print(f"  {method_label:30s}: {status} "
                      f"Iter={results['iterations']:3d} "
                      f"Residual={results['final_residual']:.2e} "
                      f"Time={results['cpu_time']:.4f}s")
                
            except Exception as e:
                print(f"  {method_label:30s}: Failed - {str(e)}")
    
    print("\n" + "=" * 80)
    print("Experiments completed!")
    
    return pd.DataFrame(all_results)


def generate_tables(results_df):
    """
    Generate numerical tables from experimental results.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing experimental results
    """
    print("\n\n" + "=" * 80)
    print("NUMERICAL TABLES FOR RSANME")
    print("=" * 80)
    
    # Table 1: Convergence comparison
    print("\n\nTable 1: Convergence Comparison")
    print("-" * 80)
    
    pivot_table = results_df.pivot_table(
        values='iterations',
        index='problem',
        columns='method_label',
        aggfunc='mean'
    )
    print(pivot_table.to_string())
    
    # Table 2: Final residuals
    print("\n\nTable 2: Final Residuals (||F(x)||)")
    print("-" * 80)
    
    pivot_table = results_df.pivot_table(
        values='final_residual',
        index='problem',
        columns='method_label',
        aggfunc='mean'
    )
    print(pivot_table.to_string(float_format=lambda x: f'{x:.2e}'))
    
    # Table 3: CPU Time comparison
    print("\n\nTable 3: CPU Time (seconds)")
    print("-" * 80)
    
    pivot_table = results_df.pivot_table(
        values='cpu_time',
        index='problem',
        columns='method_label',
        aggfunc='mean'
    )
    print(pivot_table.to_string(float_format=lambda x: f'{x:.4f}'))
    
    # Table 4: Success rate
    print("\n\nTable 4: Convergence Success Rate (%)")
    print("-" * 80)
    
    success_rate = results_df.groupby('method_label')['converged'].agg(
        lambda x: (x.sum() / len(x)) * 100
    )
    print(success_rate.to_string(float_format=lambda x: f'{x:.1f}'))
    
    # Table 5: Average iterations for converged cases
    print("\n\nTable 5: Average Iterations (Converged Cases Only)")
    print("-" * 80)
    
    converged_df = results_df[results_df['converged']]
    avg_iter = converged_df.groupby('method_label')['iterations'].mean()
    print(avg_iter.to_string(float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "=" * 80)


def save_results(results_df, filename='results.csv'):
    """
    Save results to CSV file.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing experimental results
    filename : str
        Name of the output file
    """
    # Drop residual history for CSV export
    export_df = results_df.drop(columns=['residual_history'])
    export_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Run all experiments
    results_df = run_experiments()
    
    # Generate tables
    generate_tables(results_df)
    
    # Save results
    save_results(results_df)
