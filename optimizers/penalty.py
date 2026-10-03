"""Quadratic penalty method for a scalar equality constraint."""

from typing import Callable

import numpy as np

from optimizers.base_alm import AugmentedLagrangianOptimizer
from utils.dataclasses import OptimizationResult


class QuadraticPenaltyMethod(AugmentedLagrangianOptimizer):
    r"""Minimize f(x) + (mu/2) (h(x)-b)^2 with increasing mu."""

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
        mu_update_factor: float = 10.0,
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
        self.mu = float(mu0)
        self.mu_update_factor = float(mu_update_factor)

    def optimize(self) -> OptimizationResult:
        x = self.x0.copy()
        objective_history = []
        violation_history = []
        mu_history = []

        self._log("Quadratic Penalty Method")
        for iteration in range(1, self.max_iter + 1):
            mu = self.mu

            def penalized(y: np.ndarray) -> float:
                residual = self._constraint_residual(y)
                return self.f(y) + 0.5 * mu * residual**2

            def grad_penalized(y: np.ndarray) -> np.ndarray:
                residual = self._constraint_residual(y)
                return self.grad_f(y) + mu * residual * self.grad_h(y)

            x = self._solve_unconstrained_subproblem(penalized, grad_penalized, x)
            objective = self.f(x)
            violation = self._constraint_violation(x)

            objective_history.append(objective)
            violation_history.append(violation)
            mu_history.append(mu)
            self._log(
                f"iter={iteration:3d} f={objective:.8e} "
                f"violation={violation:.3e} mu={mu:.3e}"
            )

            if violation < self.tol:
                break

            self.mu *= self.mu_update_factor
            self.subproblem_tol = max(self.tol, 0.1 * self.subproblem_tol)

        return OptimizationResult(
            solution=x,
            optimal_value=self.f(x),
            iterations=iteration,
            convergence_history=objective_history,
            constraint_violation_history=violation_history,
            method_name="Quadratic Penalty Method",
            final_constraint_violation=violation_history[-1],
            mu_history=mu_history,
        )
