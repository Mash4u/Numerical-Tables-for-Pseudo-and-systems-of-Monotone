"""
Relaxed Self-Adaptive Newton Method with Error (RSANME)

Implementation of the RSANME algorithm for solving nonlinear equations
and systems of monotone operators.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict

# Algorithm constants
MAX_LINE_SEARCH_ITERATIONS = 20


class RSANME:
    """
    Relaxed Self-Adaptive Newton Method with Error (RSANME)
    
    This class implements the RSANME algorithm for solving nonlinear equations
    of the form F(x) = 0, where F: R^n -> R^n.
    """
    
    def __init__(self, 
                 func: Callable,
                 jacobian: Callable,
                 x0: np.ndarray,
                 tol: float = 1e-6,
                 max_iter: int = 100,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.9):
        """
        Initialize RSANME solver.
        
        Parameters:
        -----------
        func : callable
            The function F(x) to find roots of
        jacobian : callable
            The Jacobian of F(x)
        x0 : ndarray
            Initial guess
        tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations
        alpha : float
            Relaxation parameter (0 < alpha < 1)
        beta : float
            Backtracking line search parameter
        gamma : float
            Armijo condition parameter
        """
        self.func = func
        self.jacobian = jacobian
        self.x0 = np.array(x0, dtype=float)
        self.tol = tol
        self.max_iter = max_iter
        
        # Validate parameters
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not (0 < gamma < 1):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def solve(self) -> Tuple[np.ndarray, Dict]:
        """
        Solve the nonlinear system using RSANME.
        
        Returns:
        --------
        x : ndarray
            Approximate solution
        info : dict
            Dictionary containing convergence information
        """
        x = self.x0.copy()
        history = {
            'iterations': [],
            'residuals': [],
            'errors': [],
            'step_sizes': []
        }
        
        for k in range(self.max_iter):
            # Evaluate function and Jacobian
            F = self.func(x)
            J = self.jacobian(x)
            
            # Compute residual norm
            residual = np.linalg.norm(F)
            history['residuals'].append(residual)
            
            # Check convergence
            if residual < self.tol:
                history['converged'] = True
                history['num_iterations'] = k
                return x, history
            
            # Solve Newton direction
            try:
                d = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # If Jacobian is singular, use pseudo-inverse
                d = -np.linalg.lstsq(J, F, rcond=None)[0]
            
            # Self-adaptive step size with backtracking line search
            t = 1.0
            for _ in range(MAX_LINE_SEARCH_ITERATIONS):
                x_new = x + self.alpha * t * d
                F_new = self.func(x_new)
                residual_new = np.linalg.norm(F_new)
                
                # Sufficient decrease condition
                if residual_new <= (1 - self.gamma * self.alpha * t) * residual:
                    break
                
                t *= self.beta
            
            history['step_sizes'].append(t)
            
            # Update solution
            x = x_new
            
            # Store iteration info
            history['iterations'].append(k)
        
        history['converged'] = False
        history['num_iterations'] = self.max_iter
        return x, history
    
    def compute_convergence_rate(self, errors: list) -> float:
        """
        Compute the convergence rate (order of convergence).
        
        Uses the formula: p ≈ log(||e_k|| / ||e_{k-1}||) / log(||e_{k-1}|| / ||e_{k-2}||)
        where e_k is the error at iteration k.
        
        Parameters:
        -----------
        errors : list
            List of error norms at each iteration
            
        Returns:
        --------
        rate : float
            Estimated order of convergence (e.g., 2 for quadratic convergence)
        """
        if len(errors) < 3:
            return 0.0
        
        # Compute convergence rates from consecutive error ratios
        rates = []
        for i in range(2, len(errors)):
            if errors[i] > 0 and errors[i-1] > 0 and errors[i-2] > 0:
                ratio_curr = errors[i] / errors[i-1]
                ratio_prev = errors[i-1] / errors[i-2]
                if ratio_prev > 0 and ratio_prev != 1.0:
                    rate = np.log(ratio_curr) / np.log(ratio_prev)
                    if 0 < rate < 10:  # Filter out unrealistic values
                        rates.append(rate)
        
        return np.mean(rates) if rates else 0.0


def modified_newton_method(func: Callable, 
                          jacobian: Callable,
                          x0: np.ndarray,
                          tol: float = 1e-6,
                          max_iter: int = 100) -> Tuple[np.ndarray, Dict]:
    """
    Classical Newton method for comparison.
    
    Parameters:
    -----------
    func : callable
        The function F(x) to find roots of
    jacobian : callable
        The Jacobian of F(x)
    x0 : ndarray
        Initial guess
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    x : ndarray
        Approximate solution
    info : dict
        Dictionary containing convergence information
    """
    x = np.array(x0, dtype=float)
    history = {
        'iterations': [],
        'residuals': [],
    }
    
    for k in range(max_iter):
        F = func(x)
        residual = np.linalg.norm(F)
        history['residuals'].append(residual)
        
        if residual < tol:
            history['converged'] = True
            history['num_iterations'] = k
            return x, history
        
        J = jacobian(x)
        try:
            d = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            d = -np.linalg.lstsq(J, F, rcond=None)[0]
        
        x = x + d
        history['iterations'].append(k)
    
    history['converged'] = False
    history['num_iterations'] = max_iter
    return x, history
