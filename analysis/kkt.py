"""KKT verification for the Q1 norm-constrained problems."""

import numpy as np

from objectives.norm_constraint import NormConstraintProblem, NormProblemType
from utils.dataclasses import KKTResult


def verify_problem_1_kkt(
    x: np.ndarray, lambda_value: float, tol: float = 1e-8
) -> KKTResult:
    """Check KKT conditions for Problem 1 at a supplied point and multiplier."""
    x = np.asarray(x, dtype=float)
    g = NormConstraintProblem.l1_constraint(x, r=1.0)
    primal_feasible = g <= tol
    dual_feasible = lambda_value >= -tol
    complementary = abs(lambda_value * g) <= tol

    # At the optimizer x=(-1,0), the first stationarity component is
    # 1 - lambda = 0.  The x2 subgradient can be chosen as zero.
    point_is_candidate = np.linalg.norm(x - np.array([-1.0, 0.0])) <= tol
    stationarity = point_is_candidate and abs(1.0 - lambda_value) <= tol

    return KKTResult(
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        complementary_slackness=complementary,
        stationarity=stationarity,
        details={"l1_constraint": float(g), "lambda_l1": float(lambda_value)},
    )


def verify_problem_2_kkt(
    x: np.ndarray, lambda_l1: float, tol: float = 1e-8
) -> KKTResult:
    r"""Check KKT feasibility at the Problem 2 optimizer.

    At x*=(-1,0), the active lower box bound x1 >= -1 contributes a
    non-negative normal-cone multiplier nu = 1-lambda_l1.  Therefore any
    lambda_l1 in [0,1] can satisfy stationarity.
    """
    x = np.asarray(x, dtype=float)
    g_l1 = NormConstraintProblem.l1_constraint(x, r=1.0)
    box_violation = max(float(np.max(np.abs(x)) - 1.0), 0.0)
    nu_lower_x1 = 1.0 - float(lambda_l1)

    primal_feasible = g_l1 <= tol and box_violation <= tol
    dual_feasible = lambda_l1 >= -tol and nu_lower_x1 >= -tol
    complementary = abs(lambda_l1 * g_l1) <= tol
    point_is_candidate = np.linalg.norm(x - np.array([-1.0, 0.0])) <= tol
    stationarity = point_is_candidate and -tol <= lambda_l1 <= 1.0 + tol

    return KKTResult(
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        complementary_slackness=complementary,
        stationarity=stationarity,
        details={
            "l1_constraint": float(g_l1),
            "lambda_l1": float(lambda_l1),
            "nu_lower_x1": nu_lower_x1,
            "box_violation": box_violation,
        },
    )


def verify_kkt(
    problem_type: NormProblemType,
    x: np.ndarray,
    lambda_value: float,
    tol: float = 1e-8,
) -> KKTResult:
    """Dispatch KKT verification for either Q1 problem."""
    if problem_type is NormProblemType.UNBOUNDED_DOMAIN:
        return verify_problem_1_kkt(x, lambda_value, tol)
    if problem_type is NormProblemType.BOX_DOMAIN:
        return verify_problem_2_kkt(x, lambda_value, tol)
    raise ValueError(f"Unknown problem type: {problem_type}")
