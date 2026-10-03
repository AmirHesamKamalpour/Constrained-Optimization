"""Q1: norm-constrained convex optimization problems."""

from enum import Enum
from typing import Tuple

import numpy as np


class NormProblemType(Enum):
    """Variants of the norm-constrained problem studied in Question 1."""

    UNBOUNDED_DOMAIN = "problem_1"
    BOX_DOMAIN = "problem_2"


class NormConstraintProblem:
    r"""Two related convex problems with an :math:`\ell_1`-ball constraint.

    Problem 1
        minimize x_1
        subject to |x_1| + |x_2| <= r, x in R^2.

    Problem 2
        minimize x_1
        subject to |x_1| + |x_2| <= r,
        |x_1| <= 1, |x_2| <= 1.
    """

    @staticmethod
    def objective(x: np.ndarray) -> float:
        """Return f(x) = x_1."""
        return float(np.asarray(x, dtype=float)[0])

    @staticmethod
    def l1_constraint(x: np.ndarray, r: float = 1.0) -> float:
        """Return g(x) = ||x||_1 - r; feasibility requires g(x) <= 0."""
        x = np.asarray(x, dtype=float)
        return float(np.abs(x[0]) + np.abs(x[1]) - r)

    @staticmethod
    def solve_primal_problem_1(r: float = 1.0) -> Tuple[np.ndarray, float]:
        """Analytical solution for Problem 1."""
        if r < 0:
            raise ValueError("r must be non-negative")
        x_star = np.array([-r, 0.0], dtype=float)
        return x_star, -float(r)

    @staticmethod
    def solve_primal_problem_2(r: float = 1.0) -> Tuple[np.ndarray, float]:
        """Analytical solution for Problem 2 with box bound 1."""
        if r < 0:
            raise ValueError("r must be non-negative")
        x1_star = -min(float(r), 1.0)
        x_star = np.array([x1_star, 0.0], dtype=float)
        return x_star, x1_star

    @classmethod
    def solve_primal(
        cls, problem_type: NormProblemType, r: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Solve either Q1 primal problem analytically."""
        if problem_type is NormProblemType.UNBOUNDED_DOMAIN:
            return cls.solve_primal_problem_1(r)
        if problem_type is NormProblemType.BOX_DOMAIN:
            return cls.solve_primal_problem_2(r)
        raise ValueError(f"Unknown problem type: {problem_type}")
