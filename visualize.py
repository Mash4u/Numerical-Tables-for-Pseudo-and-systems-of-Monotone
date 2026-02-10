"""
Visualization tools for RSANME numerical experiments.

This module provides functions to visualize convergence behavior
and compare different methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rsanme import RSANME, modified_newton_method
from test_problems import get_test_problems


def plot_convergence_history(problem, methods_params, filename='convergence.png'):
    """
    Plot convergence history for different methods on a single problem.
    
    Parameters:
    -----------
    problem : TestProblem
        The test problem to solve
    methods_params : list of tuples
        List of (method_name, params, label) tuples
    filename : str
        Name of output file
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, params, label in methods_params:
        x0 = problem.initial_guess()
        
        if method_name == 'rsanme':
            solver = RSANME(
                func=problem.func,
                jacobian=problem.jacobian,
                x0=x0,
                **params
            )
            _, info = solver.solve()
        elif method_name == 'newton':
            _, info = modified_newton_method(
                func=problem.func,
                jacobian=problem.jacobian,
                x0=x0,
                **params
            )
        
        residuals = info.get('residuals', [])
        if residuals:
            plt.semilogy(residuals, marker='o', label=label)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Residual Norm ||F(x)||', fontsize=12)
    plt.title(f'Convergence History: {problem.name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {filename}")


def plot_performance_profile(results_df, filename='performance_profile.png'):
    """
    Plot performance profile comparing different methods.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing experimental results
    filename : str
        Name of output file
    """
    plt.figure(figsize=(10, 6))
    
    methods = results_df['method_label'].unique()
    
    for method in methods:
        method_results = results_df[results_df['method_label'] == method]
        iterations = method_results['iterations'].values
        
        # Compute performance profile
        tau_values = np.linspace(1, 3, 100)
        rho_values = []
        
        for tau in tau_values:
            # Count problems where this method is within tau factor of best
            count = 0
            for problem in results_df['problem'].unique():
                problem_results = results_df[results_df['problem'] == problem]
                min_iter = problem_results['iterations'].min()
                method_iter = problem_results[
                    problem_results['method_label'] == method
                ]['iterations'].values
                
                if len(method_iter) > 0 and method_iter[0] <= tau * min_iter:
                    count += 1
            
            rho = count / len(results_df['problem'].unique())
            rho_values.append(rho)
        
        plt.plot(tau_values, rho_values, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Performance Ratio τ', fontsize=12)
    plt.ylabel('Fraction of Problems ρ(τ)', fontsize=12)
    plt.title('Performance Profile', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([1, 3])
    plt.ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance profile saved to {filename}")


def create_visualizations():
    """
    Create all visualizations for the numerical experiments.
    """
    print("Creating visualizations...")
    print("=" * 80)
    
    # Plot convergence for selected problems
    problems = get_test_problems()
    
    methods_params = [
        ('rsanme', {'tol': 1e-6, 'max_iter': 100, 'alpha': 0.5}, 'RSANME (α=0.5)'),
        ('rsanme', {'tol': 1e-6, 'max_iter': 100, 'alpha': 0.7}, 'RSANME (α=0.7)'),
        ('newton', {'tol': 1e-6, 'max_iter': 100}, 'Newton'),
    ]
    
    # Create convergence plots for first few problems
    for i, problem in enumerate(problems[:3]):
        filename = f'convergence_problem_{i+1}.png'
        try:
            plot_convergence_history(problem, methods_params, filename)
        except Exception as e:
            print(f"Failed to create plot for {problem.name}: {str(e)}")
    
    # Create performance profile if results exist
    try:
        results_df = pd.read_csv('results.csv')
        plot_performance_profile(results_df, 'performance_profile.png')
    except FileNotFoundError:
        print("Results file not found. Run experiments first.")
    
    print("\n" + "=" * 80)
    print("Visualizations completed!")


if __name__ == "__main__":
    create_visualizations()
