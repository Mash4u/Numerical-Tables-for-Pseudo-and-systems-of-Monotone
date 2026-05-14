"""
Test problems for numerical experiments with RSANME.

This module provides various test problems including:
- Nonlinear systems of equations
- Monotone operator problems
- Benchmark functions from literature
"""

import numpy as np
from typing import Tuple, Callable


class TestProblem:
    """Base class for test problems."""
    
    def __init__(self, n: int):
        self.n = n
        self.name = "Unknown"
        
    def func(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def initial_guess(self) -> np.ndarray:
        raise NotImplementedError
        
    def exact_solution(self) -> np.ndarray:
        """Return exact solution if known, otherwise None."""
        return None


class Problem1(TestProblem):
    """
    Problem 1: Nonlinear system from Broyden
    F1(x) = 3x1 - cos(x2*x3) - 1/2
    F2(x) = x1^2 - 81(x2+0.1)^2 + sin(x3) + 1.06
    F3(x) = exp(-x1*x2) + 20*x3 + (10*pi - 3)/3
    """
    
    def __init__(self):
        super().__init__(3)
        self.name = "Broyden System"
        
    def func(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros(3)
        F[0] = 3*x[0] - np.cos(x[1]*x[2]) - 0.5
        F[1] = x[0]**2 - 81*(x[1]+0.1)**2 + np.sin(x[2]) + 1.06
        F[2] = np.exp(-x[0]*x[1]) + 20*x[2] + (10*np.pi - 3)/3
        return F
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        J = np.zeros((3, 3))
        J[0, 0] = 3
        J[0, 1] = x[2] * np.sin(x[1]*x[2])
        J[0, 2] = x[1] * np.sin(x[1]*x[2])
        J[1, 0] = 2*x[0]
        J[1, 1] = -162*(x[1]+0.1)
        J[1, 2] = np.cos(x[2])
        J[2, 0] = -x[1] * np.exp(-x[0]*x[1])
        J[2, 1] = -x[0] * np.exp(-x[0]*x[1])
        J[2, 2] = 20
        return J
    
    def initial_guess(self) -> np.ndarray:
        return np.array([0.1, 0.1, -0.1])
    
    def exact_solution(self) -> np.ndarray:
        return np.array([0.5, 0.0, -0.52359877])


class Problem2(TestProblem):
    """
    Problem 2: Rosenbrock function gradient
    F(x) = grad f(x), where f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]
    """
    
    def __init__(self, n: int = 10):
        super().__init__(n)
        self.name = f"Rosenbrock Gradient (n={n})"
        
    def func(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros(self.n)
        for i in range(self.n - 1):
            F[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
            F[i+1] += 200*(x[i+1] - x[i]**2)
        return F
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        J = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            J[i, i] += -400*(x[i+1] - 3*x[i]**2) + 2
            J[i, i+1] += -400*x[i]
            J[i+1, i] += -400*x[i]
            J[i+1, i+1] += 200
        return J
    
    def initial_guess(self) -> np.ndarray:
        return -1.2 * np.ones(self.n)
    
    def exact_solution(self) -> np.ndarray:
        return np.ones(self.n)


class Problem3(TestProblem):
    """
    Problem 3: Exponential system
    F_i(x) = exp(x_i) - 1, for i = 1, ..., n
    """
    
    def __init__(self, n: int = 5):
        super().__init__(n)
        self.name = f"Exponential System (n={n})"
        
    def func(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) - 1
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.diag(np.exp(x))
    
    def initial_guess(self) -> np.ndarray:
        return np.ones(self.n) * 0.5
    
    def exact_solution(self) -> np.ndarray:
        return np.zeros(self.n)


class Problem4(TestProblem):
    """
    Problem 4: Tridiagonal system
    F_i(x) = (3 - 2*x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1
    """
    
    def __init__(self, n: int = 10):
        super().__init__(n)
        self.name = f"Tridiagonal System (n={n})"
        
    def func(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros(self.n)
        for i in range(self.n):
            F[i] = (3 - 2*x[i])*x[i] + 1
            if i > 0:
                F[i] -= x[i-1]
            if i < self.n - 1:
                F[i] -= 2*x[i+1]
        return F
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        J = np.zeros((self.n, self.n))
        for i in range(self.n):
            J[i, i] = 3 - 4*x[i]
            if i > 0:
                J[i, i-1] = -1
            if i < self.n - 1:
                J[i, i+1] = -2
        return J
    
    def initial_guess(self) -> np.ndarray:
        return -np.ones(self.n)


class Problem5(TestProblem):
    """
    Problem 5: Discrete boundary value problem
    F_i(x) = 2*x_i - x_{i-1} - x_{i+1} + h^2*(x_i + t_i + 1)^3
    where t_i = i*h, h = 1/(n+1)
    """
    
    def __init__(self, n: int = 10):
        super().__init__(n)
        self.name = f"Discrete BVP (n={n})"
        self.h = 1.0 / (n + 1)
        self.t = np.array([(i+1)*self.h for i in range(n)])
        
    def func(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros(self.n)
        h2 = self.h**2
        for i in range(self.n):
            F[i] = 2*x[i] + h2*(x[i] + self.t[i] + 1)**3
            if i > 0:
                F[i] -= x[i-1]
            if i < self.n - 1:
                F[i] -= x[i+1]
        return F
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        J = np.zeros((self.n, self.n))
        h2 = self.h**2
        for i in range(self.n):
            J[i, i] = 2 + 3*h2*(x[i] + self.t[i] + 1)**2
            if i > 0:
                J[i, i-1] = -1
            if i < self.n - 1:
                J[i, i+1] = -1
        return J
    
    def initial_guess(self) -> np.ndarray:
        return np.zeros(self.n)


def get_test_problems():
    """Return a list of all test problems."""
    return [
        Problem1(),
        Problem2(10),
        Problem2(20),
        Problem3(5),
        Problem3(10),
        Problem4(10),
        Problem4(20),
        Problem5(10),
        Problem5(20),
    ]
