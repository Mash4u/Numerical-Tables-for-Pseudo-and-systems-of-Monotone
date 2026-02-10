#!/usr/bin/env python
"""
Generate comprehensive report for RSANME numerical experiments.

This script runs all experiments, creates visualizations, and generates
a formatted report with numerical tables.
"""

import sys
from run_experiments import run_experiments, generate_tables, save_results
from visualize import create_visualizations


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def main():
    """Generate complete report."""
    print_header("RSANME Numerical Experiments Report Generator")
    
    print("This script will:")
    print("  1. Run numerical experiments on all test problems")
    print("  2. Generate numerical tables comparing methods")
    print("  3. Create convergence plots and performance profiles")
    print("  4. Save all results to files")
    print("\nThis may take a few minutes...\n")
    
    # Step 1: Run experiments
    print_header("Step 1: Running Experiments")
    try:
        results_df = run_experiments()
        print("\n✓ Experiments completed successfully")
    except Exception as e:
        print(f"\n✗ Error running experiments: {str(e)}")
        return 1
    
    # Step 2: Generate tables
    print_header("Step 2: Generating Tables")
    try:
        generate_tables(results_df)
        print("\n✓ Tables generated successfully")
    except Exception as e:
        print(f"\n✗ Error generating tables: {str(e)}")
        return 1
    
    # Step 3: Save results
    print_header("Step 3: Saving Results")
    try:
        save_results(results_df)
        print("✓ Results saved successfully")
    except Exception as e:
        print(f"\n✗ Error saving results: {str(e)}")
        return 1
    
    # Step 4: Create visualizations
    print_header("Step 4: Creating Visualizations")
    try:
        create_visualizations()
        print("\n✓ Visualizations created successfully")
    except Exception as e:
        print(f"\n✗ Error creating visualizations: {str(e)}")
        return 1
    
    # Summary
    print_header("Report Generation Complete")
    print("Generated files:")
    print("  - results.csv: Numerical results in CSV format")
    print("  - convergence_problem_*.png: Convergence plots")
    print("  - performance_profile.png: Performance comparison")
    print("\nYou can now:")
    print("  - View results.csv in Excel or any spreadsheet software")
    print("  - View the generated PNG files for visualizations")
    print("  - Use the data for your research or publications")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
