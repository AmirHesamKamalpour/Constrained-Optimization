"""Q2: equality-constrained quadratic optimization problem."""

from typing import Tuple

import numpy as np


class QuadraticEqualityProblem:
    r"""Problem used to compare penalty and multiplier methods.

    Minimize
        f(x) = x_1^2 + 2 x_2^2 + 3 x_3^2
    subject to
        x_1 + x_2 + x_3 = 1.
    """

    dimension = 3
    constraint_target = 1.0

    @staticmethod
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x[0] ** 2 + 2.0 * x[1] ** 2 + 3.0 * x[2] ** 2)

    @staticmethod
    def gradient(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.array([2.0 * x[0], 4.0 * x[1], 6.0 * x[2]])

    @staticmethod
    def constraint(x: np.ndarray) -> float:
        """Return h(x) = x_1 + x_2 + x_3."""
        return float(np.sum(np.asarray(x, dtype=float)))

    @staticmethod
    def constraint_gradient(x: np.ndarray) -> np.ndarray:
        """Return grad h(x) = [1, 1, 1]."""
        del x
        return np.ones(3)

    @classmethod
    def residual(cls, x: np.ndarray) -> float:
        """Return h(x) - 1."""
        return cls.constraint(x) - cls.constraint_target

    @classmethod
    def analytical_solution(cls) -> Tuple[np.ndarray, float, float]:
        """Return (x*, f*, lambda*) from the KKT equations."""
        lambda_star = -12.0 / 11.0
        x_star = np.array([6.0 / 11.0, 3.0 / 11.0, 2.0 / 11.0])
        return x_star, cls.objective(x_star), lambda_star
