"""Base class for equality-constrained penalty and ALM methods."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from optimizers.unconstrained import bfgs_minimize
from utils.dataclasses import OptimizationResult

ArrayFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]


class AugmentedLagrangianOptimizer(ABC):
    """Common machinery for scalar equality-constrained optimization."""

    def __init__(
        self,
        objective_func: ArrayFunction,
        constraint_func: ArrayFunction,
        gradient_func: GradientFunction,
        constraint_grad_func: GradientFunction,
        initial_point: np.ndarray,
        constraint_target: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100,
        subproblem_tol: float = 1e-4,
        max_sub_iter: int = 100,
        verbose: bool = False,
    ) -> None:
        self.f = objective_func
        self.h = constraint_func
        self.grad_f = gradient_func
        self.grad_h = constraint_grad_func
        self.x0 = np.asarray(initial_point, dtype=float).copy()
        self.b = float(constraint_target)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.subproblem_tol = float(subproblem_tol)
        self.max_sub_iter = int(max_sub_iter)
        self.verbose = bool(verbose)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.x0.ndim != 1:
            raise ValueError("initial_point must be a one-dimensional array")
        if self.tol <= 0 or self.subproblem_tol <= 0:
            raise ValueError("tolerances must be positive")
        if self.max_iter <= 0 or self.max_sub_iter <= 0:
            raise ValueError("iteration limits must be positive")

    def _solve_unconstrained_subproblem(
        self,
        func: ArrayFunction,
        grad_func: GradientFunction,
        x_init: np.ndarray,
    ) -> np.ndarray:
        result = bfgs_minimize(
            func,
            grad_func,
            x_init,
            tol=self.subproblem_tol,
            max_iter=self.max_sub_iter,
        )
        return result.solution

    def _constraint_residual(self, x: np.ndarray) -> float:
        return float(self.h(x) - self.b)

    def _constraint_violation(self, x: np.ndarray) -> float:
        return abs(self._constraint_residual(x))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run the optimization method."""
