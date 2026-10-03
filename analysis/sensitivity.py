"""Optimal-value and sensitivity calculations for Q1."""

from typing import Dict

import numpy as np

from objectives.norm_constraint import NormProblemType


def optimal_value(problem_type: NormProblemType, r: np.ndarray | float) -> np.ndarray:
    """Evaluate the optimal value function V(r) for either Q1 problem."""
    r_values = np.asarray(r, dtype=float)
    if np.any(r_values < 0):
        raise ValueError("r must be non-negative")

    if problem_type is NormProblemType.UNBOUNDED_DOMAIN:
        return -r_values
    if problem_type is NormProblemType.BOX_DOMAIN:
        return -np.minimum(r_values, 1.0)
    raise ValueError(f"Unknown problem type: {problem_type}")


def finite_difference_sensitivity(
    problem_type: NormProblemType,
    r: float = 1.0,
    epsilon: float = 1e-2,
) -> Dict[str, float]:
    """Compute one-sided finite-difference slopes of V(r)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if r - epsilon < 0:
        raise ValueError("r - epsilon must remain non-negative")

    center = float(optimal_value(problem_type, r))
    left = float(optimal_value(problem_type, r - epsilon))
    right = float(optimal_value(problem_type, r + epsilon))

    return {
        "left_derivative": (center - left) / epsilon,
        "right_derivative": (right - center) / epsilon,
    }


def theoretical_sensitivity_at_one(problem_type: NormProblemType) -> Dict[str, float]:
    """Return the theoretical one-sided slopes of V(r) at r=1."""
    if problem_type is NormProblemType.UNBOUNDED_DOMAIN:
        return {"left_derivative": -1.0, "right_derivative": -1.0}
    if problem_type is NormProblemType.BOX_DOMAIN:
        return {"left_derivative": -1.0, "right_derivative": 0.0}
    raise ValueError(f"Unknown problem type: {problem_type}")
