"""Shared data containers for optimization and analysis results."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class UnconstrainedResult:
    """Result of an unconstrained BFGS subproblem."""

    solution: np.ndarray
    objective_value: float
    iterations: int
    gradient_norm: float
    converged: bool


@dataclass
class OptimizationResult:
    """Result returned by the constrained optimization methods."""

    solution: np.ndarray
    optimal_value: float
    iterations: int
    convergence_history: List[float]
    constraint_violation_history: List[float]
    method_name: str
    final_constraint_violation: float
    lambda_history: Optional[List[float]] = None
    mu_history: List[float] = field(default_factory=list)


@dataclass
class KKTResult:
    """Summary of KKT checks for the Q1 norm-constrained problems."""

    primal_feasible: bool
    dual_feasible: bool
    complementary_slackness: bool
    stationarity: bool
    details: Dict[str, float] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return (
            self.primal_feasible
            and self.dual_feasible
            and self.complementary_slackness
            and self.stationarity
        )
