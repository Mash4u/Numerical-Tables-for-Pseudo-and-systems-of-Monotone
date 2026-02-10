"""
Relaxed Self-Adaptive Newton Method with Error (RSANME)

Implementation of the RSANME algorithm for solving nonlinear equations
and systems of monotone operators.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict


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
            max_line_search_iter = 20
            for _ in range(max_line_search_iter):
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
        Compute the convergence rate.
        
        Parameters:
        -----------
        errors : list
            List of error norms
            
        Returns:
        --------
        rate : float
            Estimated convergence rate
        """
        if len(errors) < 3:
            return 0.0
        
        # Compute convergence rate using last few iterations
        rates = []
        for i in range(len(errors) - 2, len(errors) - 1):
            if errors[i] > 0 and errors[i-1] > 0:
                rate = np.log(errors[i]) / np.log(errors[i-1])
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
