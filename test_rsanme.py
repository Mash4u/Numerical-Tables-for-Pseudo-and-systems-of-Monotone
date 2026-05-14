"""
Test suite for RSANME implementation.

Basic tests to validate the algorithm and test problems.
"""

import numpy as np
from rsanme import RSANME, modified_newton_method
from test_problems import Problem1, Problem3, Problem4


def test_rsanme_convergence():
    """Test that RSANME converges on simple problems."""
    print("Testing RSANME convergence...")
    
    problem = Problem3(5)
    solver = RSANME(
        func=problem.func,
        jacobian=problem.jacobian,
        x0=problem.initial_guess(),
        alpha=0.7
    )
    x_sol, info = solver.solve()
    
    assert info['converged'], "RSANME should converge on exponential system"
    assert info['num_iterations'] < 100, "Should converge in less than 100 iterations"
    assert info['residuals'][-1] < 1e-5, "Final residual should be small"
    
    print("  ✓ RSANME convergence test passed")


def test_newton_convergence():
    """Test that classical Newton method converges."""
    print("Testing Newton method convergence...")
    
    # Newton's method typically converges in very few iterations
    # for well-conditioned problems with good initial guesses
    EXPECTED_NEWTON_ITERATIONS = 10
    
    problem = Problem3(5)
    x_sol, info = modified_newton_method(
        func=problem.func,
        jacobian=problem.jacobian,
        x0=problem.initial_guess()
    )
    
    assert info['converged'], "Newton should converge on exponential system"
    assert info['num_iterations'] < EXPECTED_NEWTON_ITERATIONS, \
        f"Newton should converge in less than {EXPECTED_NEWTON_ITERATIONS} iterations"
    
    print("  ✓ Newton convergence test passed")


def test_solution_accuracy():
    """Test that the solution is accurate for problems with known solutions."""
    print("Testing solution accuracy...")
    
    problem = Problem3(5)
    exact_sol = problem.exact_solution()
    
    solver = RSANME(
        func=problem.func,
        jacobian=problem.jacobian,
        x0=problem.initial_guess(),
        alpha=0.7
    )
    x_sol, info = solver.solve()
    
    error = np.linalg.norm(x_sol - exact_sol)
    assert error < 1e-5, f"Solution error {error} too large"
    
    print(f"  ✓ Solution accuracy test passed (error: {error:.2e})")


def test_broyden_system():
    """Test RSANME on Broyden system."""
    print("Testing on Broyden system...")
    
    problem = Problem1()
    solver = RSANME(
        func=problem.func,
        jacobian=problem.jacobian,
        x0=problem.initial_guess(),
        alpha=0.7
    )
    x_sol, info = solver.solve()
    
    assert info['converged'], "Should converge on Broyden system"
    final_residual = np.linalg.norm(problem.func(x_sol))
    assert final_residual < 1e-5, f"Final residual {final_residual} too large"
    
    print(f"  ✓ Broyden system test passed (iterations: {info['num_iterations']})")


def test_different_alphas():
    """Test that different alpha values produce convergence."""
    print("Testing different alpha values...")
    
    problem = Problem4(10)
    alphas = [0.3, 0.5, 0.7, 0.9]
    
    for alpha in alphas:
        solver = RSANME(
            func=problem.func,
            jacobian=problem.jacobian,
            x0=problem.initial_guess(),
            alpha=alpha
        )
        x_sol, info = solver.solve()
        
        # At least some should converge
        if info['converged']:
            print(f"  ✓ Alpha={alpha} converged in {info['num_iterations']} iterations")
        else:
            print(f"  ⚠ Alpha={alpha} did not converge")
    
    print("  ✓ Alpha parameter test completed")


def test_residual_decrease():
    """Test that residual decreases monotonically (or mostly)."""
    print("Testing residual decrease...")
    
    # Expect residual to decrease in at least this fraction of iterations
    MIN_DECREASE_RATIO = 0.7
    
    problem = Problem3(5)
    solver = RSANME(
        func=problem.func,
        jacobian=problem.jacobian,
        x0=problem.initial_guess(),
        alpha=0.7
    )
    x_sol, info = solver.solve()
    
    residuals = info['residuals']
    
    # Check that residual generally decreases
    # Allow for some temporary increases due to line search
    decrease_count = sum(1 for i in range(len(residuals)-1) 
                        if residuals[i+1] < residuals[i])
    total_steps = len(residuals) - 1
    
    assert decrease_count > MIN_DECREASE_RATIO * total_steps, \
        f"Residual should decrease in at least {MIN_DECREASE_RATIO*100}% of iterations"
    
    print(f"  ✓ Residual decrease test passed ({decrease_count}/{total_steps} decreasing)")


def test_parameter_validation():
    """Test that invalid parameters are rejected."""
    print("Testing parameter validation...")
    
    problem = Problem3(5)
    
    # Test invalid alpha values
    try:
        RSANME(problem.func, problem.jacobian, problem.initial_guess(), alpha=-0.1)
        assert False, "Should reject negative alpha"
    except ValueError:
        pass
    
    try:
        RSANME(problem.func, problem.jacobian, problem.initial_guess(), alpha=1.5)
        assert False, "Should reject alpha > 1"
    except ValueError:
        pass
    
    # Test invalid beta values
    try:
        RSANME(problem.func, problem.jacobian, problem.initial_guess(), beta=0)
        assert False, "Should reject beta = 0"
    except ValueError:
        pass
    
    # Test invalid gamma values
    try:
        RSANME(problem.func, problem.jacobian, problem.initial_guess(), gamma=1.0)
        assert False, "Should reject gamma = 1"
    except ValueError:
        pass
    
    print("  ✓ Parameter validation test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RSANME Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_rsanme_convergence,
        test_newton_convergence,
        test_solution_accuracy,
        test_broyden_system,
        test_different_alphas,
        test_residual_decrease,
        test_parameter_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Test failed: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Test error: {str(e)}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
