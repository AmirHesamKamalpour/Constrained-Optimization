"""Plotting helpers for Q1 analysis and Q2 method comparison."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.duality import dual_function_problem_1, dual_function_problem_2
from analysis.sensitivity import optimal_value
from objectives.norm_constraint import NormProblemType
from objectives.quadratic_eq import QuadraticEqualityProblem
from utils.dataclasses import OptimizationResult


def _finish_figure(fig, save_path: Optional[str | Path], show: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_q1_value_and_feasible_regions(
    r_min: float = 0.0,
    r_max: float = 2.0,
    num_points: int = 500,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot Q1 value functions, sensitivities, and feasible regions."""
    r_values = np.linspace(r_min, r_max, num_points)
    value_1 = optimal_value(NormProblemType.UNBOUNDED_DOMAIN, r_values)
    value_2 = optimal_value(NormProblemType.BOX_DOMAIN, r_values)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(r_values, value_1, label=r"$V_1(r)=-r$")
    axes[0, 0].axvline(1.0, linestyle="--", alpha=0.5)
    axes[0, 0].axhline(-1.0, linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Problem 1 optimal value")
    axes[0, 0].set_xlabel("constraint radius r")
    axes[0, 0].set_ylabel("V(r)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(r_values, value_2, label=r"$V_2(r)=-\min(r,1)$")
    axes[0, 1].axvline(1.0, linestyle="--", alpha=0.5)
    axes[0, 1].axhline(-1.0, linestyle="--", alpha=0.5)
    axes[0, 1].set_title("Problem 2 optimal value")
    axes[0, 1].set_xlabel("constraint radius r")
    axes[0, 1].set_ylabel("V(r)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    derivative_1 = -np.ones_like(r_values)
    derivative_2 = np.where(r_values < 1.0, -1.0, 0.0)
    axes[1, 0].plot(r_values, derivative_1, label="Problem 1")
    axes[1, 0].plot(r_values, derivative_2, linestyle="--", label="Problem 2")
    axes[1, 0].axvline(1.0, linestyle=":", alpha=0.5)
    axes[1, 0].set_title("Sensitivity of the optimal value")
    axes[1, 0].set_xlabel("constraint radius r")
    axes[1, 0].set_ylabel("dV/dr")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    diamond_x = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    diamond_y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    box_x = np.array([-1.0, 1.0, 1.0, -1.0, -1.0])
    box_y = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    axes[1, 1].fill(diamond_x, diamond_y, alpha=0.2, label=r"$\|x\|_1\leq1$")
    axes[1, 1].plot(box_x, box_y, linestyle="--", label="box boundary")
    axes[1, 1].scatter([-1.0], [0.0], s=70, label=r"$x^*=(-1,0)$")
    axes[1, 1].set_title("Feasible sets at r = 1")
    axes[1, 1].set_xlabel(r"$x_1$")
    axes[1, 1].set_ylabel(r"$x_2$")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlim(-1.4, 1.4)
    axes[1, 1].set_ylim(-1.4, 1.4)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    _finish_figure(fig, save_path, show)


def plot_q1_dual_functions(
    lambda_max: float = 2.0,
    num_points: int = 500,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot the two Q1 dual functions."""
    lambda_values = np.linspace(0.0, lambda_max, num_points)
    q1 = np.array([dual_function_problem_1(value) for value in lambda_values])
    q2 = np.array([dual_function_problem_2(value) for value in lambda_values])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(lambda_values, q1)
    axes[0].scatter([1.0], [-1.0], zorder=3, label=r"$\lambda^*=1$")
    axes[0].set_title("Problem 1 dual function")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$q(\lambda)$")
    axes[0].set_ylim(-2.1, -0.8)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(lambda_values, q2)
    axes[1].axvspan(0.0, 1.0, alpha=0.15, label=r"optimal $\lambda\in[0,1]$")
    axes[1].set_title("Problem 2 dual function")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$q(\lambda)$")
    axes[1].set_ylim(-2.1, -0.8)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    _finish_figure(fig, save_path, show)


def plot_q2_comparison(
    penalty_result: OptimizationResult,
    multiplier_result: OptimizationResult,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Visualize Q2 convergence, solution error, and multiplier behavior."""
    x_star, f_star, lambda_star = QuadraticEqualityProblem.analytical_solution()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    axes[0, 0].plot(penalty_result.convergence_history, label="Penalty")
    axes[0, 0].plot(multiplier_result.convergence_history, linestyle="--", label="Multipliers")
    axes[0, 0].axhline(f_star, linestyle=":", label="Analytical f*")
    axes[0, 0].set_title("Objective convergence")
    axes[0, 0].set_xlabel("outer iteration")
    axes[0, 0].set_ylabel("f(x)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].semilogy(penalty_result.constraint_violation_history, label="Penalty")
    axes[0, 1].semilogy(multiplier_result.constraint_violation_history, linestyle="--", label="Multipliers")
    axes[0, 1].set_title("Constraint violation")
    axes[0, 1].set_xlabel("outer iteration")
    axes[0, 1].set_ylabel(r"$|h(x)-1|$")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    indices = np.arange(3)
    width = 0.25
    axes[0, 2].bar(indices - width, x_star, width, label="Analytical")
    axes[0, 2].bar(indices, penalty_result.solution, width, label="Penalty")
    axes[0, 2].bar(indices + width, multiplier_result.solution, width, label="Multipliers")
    axes[0, 2].set_title("Final solution")
    axes[0, 2].set_xticks(indices, [r"$x_1$", r"$x_2$", r"$x_3$"])
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend()

    if multiplier_result.lambda_history:
        axes[1, 0].plot(multiplier_result.lambda_history, label=r"$\lambda_k$")
        axes[1, 0].axhline(lambda_star, linestyle="--", label=r"analytical $\lambda^*$")
    axes[1, 0].set_title("Multiplier convergence")
    axes[1, 0].set_xlabel("outer iteration")
    axes[1, 0].set_ylabel(r"$\lambda$")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].semilogy(penalty_result.mu_history, label="Penalty")
    axes[1, 1].semilogy(multiplier_result.mu_history, linestyle="--", label="Multipliers")
    axes[1, 1].set_title("Penalty parameter")
    axes[1, 1].set_xlabel("outer iteration")
    axes[1, 1].set_ylabel(r"$\mu$")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].axis("off")
    text = (
        "Analytical solution\n"
        f"x* = {np.array2string(x_star, precision=6)}\n"
        f"f* = {f_star:.10f}\n\n"
        "Penalty method\n"
        f"iterations = {penalty_result.iterations}\n"
        f"violation = {penalty_result.final_constraint_violation:.2e}\n\n"
        "Method of multipliers\n"
        f"iterations = {multiplier_result.iterations}\n"
        f"violation = {multiplier_result.final_constraint_violation:.2e}"
    )
    axes[1, 2].text(0.02, 0.98, text, va="top", family="monospace")

    _finish_figure(fig, save_path, show)
