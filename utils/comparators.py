"""Comparison helpers for the equality-constrained Q2 experiment."""

from typing import Dict, Iterable, Tuple

import numpy as np

from objectives.quadratic_eq import QuadraticEqualityProblem
from optimizers.multipliers import MethodOfMultipliers
from optimizers.penalty import QuadraticPenaltyMethod
from utils.dataclasses import OptimizationResult


def build_q2_optimizers(
    initial_point: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[QuadraticPenaltyMethod, MethodOfMultipliers]:
    """Construct both Q2 methods with the notebook's hyperparameters."""
    p = QuadraticEqualityProblem
    common = dict(
        objective_func=p.objective,
        constraint_func=p.constraint,
        gradient_func=p.gradient,
        constraint_grad_func=p.constraint_gradient,
        initial_point=np.asarray(initial_point, dtype=float),
        constraint_target=p.constraint_target,
        tol=tol,
        max_iter=max_iter,
        mu0=1.0,
        mu_update_factor=10.0,
        verbose=verbose,
    )
    return QuadraticPenaltyMethod(**common), MethodOfMultipliers(lambda0=0.0, **common)


def compare_q2_methods(
    initial_point: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[OptimizationResult, OptimizationResult]:
    """Run the quadratic penalty and method-of-multipliers algorithms."""
    if initial_point is None:
        initial_point = np.array([0.1, 0.2, 0.7])
    penalty, multipliers = build_q2_optimizers(initial_point, tol, max_iter, verbose)
    return penalty.optimize(), multipliers.optimize()


def result_metrics(result: OptimizationResult) -> Dict[str, float]:
    """Compute error metrics against the Q2 analytical solution."""
    x_star, f_star, _ = QuadraticEqualityProblem.analytical_solution()
    return {
        "iterations": float(result.iterations),
        "solution_error": float(np.linalg.norm(result.solution - x_star)),
        "objective_error": abs(float(result.optimal_value) - f_star),
        "constraint_violation": float(result.final_constraint_violation),
    }


def initial_point_sensitivity(
    points: Iterable[np.ndarray],
    tol: float = 1e-6,
    max_iter: int = 50,
) -> list[dict]:
    """Run both methods from several starting points."""
    records = []
    for point in points:
        point = np.asarray(point, dtype=float)
        penalty_result, multiplier_result = compare_q2_methods(
            point, tol=tol, max_iter=max_iter, verbose=False
        )
        records.append(
            {
                "initial_point": point,
                "penalty": result_metrics(penalty_result),
                "multipliers": result_metrics(multiplier_result),
            }
        )
    return records
