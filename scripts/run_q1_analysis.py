"""Run the Question 1 primal/dual/KKT/sensitivity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.duality import (  # noqa: E402
    duality_gap,
    solve_dual_problem_1,
    solve_dual_problem_2,
)
from analysis.kkt import verify_problem_1_kkt, verify_problem_2_kkt  # noqa: E402
from analysis.sensitivity import (  # noqa: E402
    finite_difference_sensitivity,
    theoretical_sensitivity_at_one,
)
from objectives.norm_constraint import NormConstraintProblem, NormProblemType  # noqa: E402
from utils.plotting import plot_q1_dual_functions, plot_q1_value_and_feasible_regions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for generated PNG figures.",
    )
    args = parser.parse_args()

    x1, p1 = NormConstraintProblem.solve_primal_problem_1()
    lambda1, d1 = solve_dual_problem_1()
    kkt1 = verify_problem_1_kkt(x1, lambda1)

    x2, p2 = NormConstraintProblem.solve_primal_problem_2()
    lambda_interval, d2 = solve_dual_problem_2()
    kkt2_examples = {
        value: verify_problem_2_kkt(x2, value) for value in (0.0, 0.5, 1.0)
    }

    print("=" * 72)
    print("Q1 — DUALITY, KKT CONDITIONS, AND SENSITIVITY")
    print("=" * 72)
    print("\nProblem 1: min x1 subject to |x1| + |x2| <= 1, x in R^2")
    print(f"  x* = {x1}")
    print(f"  p* = {p1:.6f}")
    print(f"  lambda* = {lambda1:.6f}")
    print(f"  d* = {d1:.6f}")
    print(f"  duality gap = {duality_gap(p1, d1):.2e}")
    print(f"  KKT satisfied = {kkt1.satisfied}")

    print("\nProblem 2: same L1 constraint with |x1| <= 1 and |x2| <= 1")
    print(f"  x* = {x2}")
    print(f"  p* = {p2:.6f}")
    print(f"  optimal L1-multiplier interval = {lambda_interval}")
    print(f"  d* = {d2:.6f}")
    print(f"  duality gap = {duality_gap(p2, d2):.2e}")
    for value, check in kkt2_examples.items():
        print(
            f"  lambda={value:.1f}: KKT={check.satisfied}, "
            f"active-box multiplier={check.details['nu_lower_x1']:.1f}"
        )

    print("\nSensitivity at r = 1")
    for problem_type, label in [
        (NormProblemType.UNBOUNDED_DOMAIN, "Problem 1"),
        (NormProblemType.BOX_DOMAIN, "Problem 2"),
    ]:
        numerical = finite_difference_sensitivity(problem_type)
        theoretical = theoretical_sensitivity_at_one(problem_type)
        print(
            f"  {label}: numerical left/right = "
            f"({numerical['left_derivative']:.4f}, {numerical['right_derivative']:.4f}); "
            f"theory = ({theoretical['left_derivative']:.1f}, "
            f"{theoretical['right_derivative']:.1f})"
        )

    save_value = None
    save_dual = None
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_value = args.save_dir / "q1_value_sensitivity.png"
        save_dual = args.save_dir / "q1_dual_functions.png"

    plot_q1_value_and_feasible_regions(save_path=save_value, show=not args.no_show)
    plot_q1_dual_functions(save_path=save_dual, show=not args.no_show)


if __name__ == "__main__":
    main()
