"""Method of multipliers (augmented Lagrangian method)."""

from typing import Callable

import numpy as np

from optimizers.base_alm import AugmentedLagrangianOptimizer
from utils.dataclasses import OptimizationResult


class MethodOfMultipliers(AugmentedLagrangianOptimizer):
    r"""Augmented Lagrangian method for one scalar equality constraint."""

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        constraint_func: Callable[[np.ndarray], float],
        gradient_func: Callable[[np.ndarray], np.ndarray],
        constraint_grad_func: Callable[[np.ndarray], np.ndarray],
        initial_point: np.ndarray,
        constraint_target: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100,
        mu0: float = 1.0,
        lambda0: float = 0.0,
        mu_update_factor: float = 10.0,
        violation_reduction_threshold: float = 0.25,
        subproblem_tol: float = 1e-4,
        max_sub_iter: int = 100,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            objective_func,
            constraint_func,
            gradient_func,
            constraint_grad_func,
            initial_point,
            constraint_target,
            tol,
            max_iter,
            subproblem_tol,
            max_sub_iter,
            verbose,
        )
        if mu0 <= 0 or mu_update_factor <= 1:
            raise ValueError("mu0 must be positive and mu_update_factor must exceed 1")
        if not 0 < violation_reduction_threshold < 1:
            raise ValueError("violation_reduction_threshold must lie in (0, 1)")
        self.mu = float(mu0)
        self.lambda_value = float(lambda0)
        self.mu_update_factor = float(mu_update_factor)
        self.violation_reduction_threshold = float(violation_reduction_threshold)

    def optimize(self) -> OptimizationResult:
        x = self.x0.copy()
        objective_history = []
        violation_history = []
        lambda_history = [self.lambda_value]
        mu_history = []

        self._log("Method of Multipliers")
        for iteration in range(1, self.max_iter + 1):
            mu = self.mu
            lambda_value = self.lambda_value

            def augmented_lagrangian(y: np.ndarray) -> float:
                residual = self._constraint_residual(y)
                return self.f(y) + lambda_value * residual + 0.5 * mu * residual**2

            def grad_augmented_lagrangian(y: np.ndarray) -> np.ndarray:
                residual = self._constraint_residual(y)
                return self.grad_f(y) + (lambda_value + mu * residual) * self.grad_h(y)

            x = self._solve_unconstrained_subproblem(
                augmented_lagrangian, grad_augmented_lagrangian, x
            )
            residual = self._constraint_residual(x)
            violation = abs(residual)
            self.lambda_value += mu * residual

            objective_history.append(self.f(x))
            violation_history.append(violation)
            lambda_history.append(self.lambda_value)
            mu_history.append(mu)
            self._log(
                f"iter={iteration:3d} f={self.f(x):.8e} "
                f"violation={violation:.3e} lambda={self.lambda_value:.6e} mu={mu:.3e}"
            )

            if violation < self.tol:
                break

            if iteration > 1:
                previous = violation_history[-2]
                if violation > self.violation_reduction_threshold * previous:
                    self.mu *= self.mu_update_factor

            self.subproblem_tol = max(self.tol, 0.1 * self.subproblem_tol)

        return OptimizationResult(
            solution=x,
            optimal_value=self.f(x),
            iterations=iteration,
            convergence_history=objective_history,
            constraint_violation_history=violation_history,
            method_name="Method of Multipliers",
            final_constraint_violation=violation_history[-1],
            lambda_history=lambda_history,
            mu_history=mu_history,
        )
