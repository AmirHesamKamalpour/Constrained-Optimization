"""Dual functions and dual solutions for the Q1 norm-constrained problems."""

from typing import List, Tuple

import numpy as np

from objectives.norm_constraint import NormProblemType


def dual_function_problem_1(lambda_value: float) -> float:
    r"""Return q(lambda) for min x1 s.t. |x1| + |x2| <= 1 over R^2.

    q(lambda) = -infinity for 0 <= lambda < 1 and -lambda for lambda >= 1.
    """
    if lambda_value < 0:
        raise ValueError("lambda must be non-negative")
    if lambda_value < 1.0:
        return -np.inf
    return -float(lambda_value)


def dual_function_problem_2(lambda_value: float) -> float:
    r"""Return q(lambda) for the same L1 constraint with x in [-1,1]^2.

    q(lambda) = -1 for 0 <= lambda <= 1 and -lambda for lambda > 1.
    """
    if lambda_value < 0:
        raise ValueError("lambda must be non-negative")
    if lambda_value <= 1.0:
        return -1.0
    return -float(lambda_value)


def solve_dual_problem_1() -> Tuple[float, float]:
    """Return the unique dual solution lambda*=1 and q*=-1."""
    return 1.0, -1.0


def solve_dual_problem_2() -> Tuple[List[float], float]:
    """Return the optimal interval lambda* in [0,1] and q*=-1."""
    return [0.0, 1.0], -1.0


def dual_function(problem_type: NormProblemType, lambda_value: float) -> float:
    """Evaluate the appropriate Q1 dual function."""
    if problem_type is NormProblemType.UNBOUNDED_DOMAIN:
        return dual_function_problem_1(lambda_value)
    if problem_type is NormProblemType.BOX_DOMAIN:
        return dual_function_problem_2(lambda_value)
    raise ValueError(f"Unknown problem type: {problem_type}")


def duality_gap(primal_value: float, dual_value: float) -> float:
    """Return |p* - d*|."""
    return abs(float(primal_value) - float(dual_value))
