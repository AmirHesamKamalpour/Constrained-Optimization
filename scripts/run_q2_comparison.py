"""Run the Question 2 penalty-vs-multipliers comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objectives.quadratic_eq import QuadraticEqualityProblem  # noqa: E402
from utils.comparators import (  # noqa: E402
    compare_q2_methods,
    initial_point_sensitivity,
    result_metrics,
)
from utils.plotting import plot_q2_comparison  # noqa: E402


def _print_result(name, result) -> None:
    metrics = result_metrics(result)
    print(f"\n{name}")
    print(f"  x = {np.array2string(result.solution, precision=9)}")
    print(f"  f(x) = {result.optimal_value:.10f}")
    print(f"  outer iterations = {result.iterations}")
    print(f"  solution error = {metrics['solution_error']:.2e}")
    print(f"  constraint violation = {metrics['constraint_violation']:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print every outer iteration.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for generated PNG figures.",
    )
    args = parser.parse_args()

    x0 = np.array([0.1, 0.2, 0.7])
    x_star, f_star, lambda_star = QuadraticEqualityProblem.analytical_solution()
    penalty, multipliers = compare_q2_methods(x0, verbose=args.verbose)

    print("=" * 72)
    print("Q2 — QUADRATIC PENALTY METHOD VS METHOD OF MULTIPLIERS")
    print("=" * 72)
    print("Problem: min x1^2 + 2 x2^2 + 3 x3^2 subject to x1+x2+x3=1")
    print(f"\nAnalytical x* = {np.array2string(x_star, precision=9)}")
    print(f"Analytical f* = {f_star:.10f}")
    print(f"Analytical lambda* = {lambda_star:.10f}")

    _print_result("Quadratic penalty", penalty)
    _print_result("Method of multipliers", multipliers)

    test_points = [
        np.array([0.1, 0.2, 0.7]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
    ]
    print("\nSensitivity to initial point")
    print("  start point              penalty iters/error       multipliers iters/error")
    for record in initial_point_sensitivity(test_points):
        point = np.array2string(record["initial_point"], precision=1)
        p = record["penalty"]
        m = record["multipliers"]
        print(
            f"  {point:<24} "
            f"{int(p['iterations']):>2d} / {p['solution_error']:.2e}        "
            f"{int(m['iterations']):>2d} / {m['solution_error']:.2e}"
        )

    save_path = None
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = args.save_dir / "q2_method_comparison.png"
    plot_q2_comparison(penalty, multipliers, save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
