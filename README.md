# Constrained Optimization: Duality, Sensitivity, and Augmented Lagrangian Methods

A compact implementation and analysis of two constrained optimization studies:

1. **Duality, KKT conditions, and sensitivity** for linear minimization over an $\ell_1$-norm constraint, with and without additional box constraints.
2. **Quadratic penalty vs. method of multipliers** for an equality-constrained convex quadratic problem, using BFGS with Armijo backtracking to solve the unconstrained subproblems.

---

## Question 1 — Duality, KKT Conditions, and Sensitivity

The first study compares two closely related convex optimization problems.

### Problem 1: $\ell_1$-ball over $\mathbb{R}^2$

$$
\begin{aligned}
\min_{x \in \mathbb{R}^2} \quad & x_1 \\
\text{s.t.} \quad & |x_1| + |x_2| \le 1.
\end{aligned}
$$

The minimum occurs at the leftmost point of the unit $\ell_1$-ball:

$$
x^\star = (-1,0),
\qquad
p^\star = -1.
$$

With multiplier $\lambda \ge 0$, the Lagrangian is

$$
L(x,\lambda)
=
x_1
+
\lambda \left(|x_1| + |x_2| - 1\right).
$$

The dual function is

$$
q(\lambda)
=
\begin{cases}
-\infty, & 0 \le \lambda < 1, \\
-\lambda, & \lambda \ge 1.
\end{cases}
$$

Therefore,

$$
\lambda^\star = 1,
\qquad
d^\star = -1,
$$

and the duality gap is zero.

If the norm bound is changed from $1$ to $r$, the optimal value function is

$$
V_1(r) = -r,
$$

so its sensitivity is

$$
\frac{dV_1}{dr}
=
-1
=
-\lambda^\star.
$$

### Problem 2: Same Norm Constraint with a Box Domain

The second problem adds $|x_1| \le 1$ and $|x_2| \le 1$:

$$
\begin{aligned}
\min_x \quad & x_1 \\
\text{s.t.} \quad
& |x_1| + |x_2| \le 1, \\
& |x_1| \le 1, \qquad |x_2| \le 1.
\end{aligned}
$$

At $r=1$, the primal solution is still

$$
x^\star = (-1,0),
\qquad
p^\star = -1.
$$

The dual function associated with the $\ell_1$ constraint becomes

$$
q(\lambda)
=
\begin{cases}
-1, & 0 \le \lambda \le 1, \\
-\lambda, & \lambda > 1.
\end{cases}
$$

Hence, the optimal $\ell_1$-constraint multiplier is not unique:

$$
\lambda^\star \in [0,1].
$$

The KKT implementation makes the source of this non-uniqueness explicit. At $x_1=-1$, the lower box bound is active and contributes a normal-cone multiplier

$$
\nu = 1 - \lambda.
$$

Thus, every $\lambda \in [0,1]$ can be paired with a non-negative $\nu$ to satisfy stationarity.

For a varying norm radius $r$, the optimal value function is

$$
V_2(r)
=
-\min(r,1).
$$

It has a kink at $r=1$:

$$
V_2'(1^-)
=
-1,
\qquad
V_2'(1^+)
=
0.
$$

The numerical finite-difference check in `analysis/sensitivity.py` reproduces these one-sided derivatives.

---

## Question 2 — Penalty Method vs. Method of Multipliers

The second study solves

$$
\begin{aligned}
\min_{x \in \mathbb{R}^3} \quad
& x_1^2 + 2x_2^2 + 3x_3^2 \\
\text{s.t.} \quad
& x_1 + x_2 + x_3 = 1.
\end{aligned}
$$

The KKT equations give the analytical solution

$$
x^\star
=
\left(
\frac{6}{11},
\frac{3}{11},
\frac{2}{11}
\right),
$$

with

$$
f^\star
=
\frac{6}{11}
\approx
0.5454545455,
\qquad
\lambda^\star
=
-\frac{12}{11}.
$$

### Quadratic Penalty Method

The penalty method solves a sequence of unconstrained problems

$$
\min_x
\;
f(x)
+
\frac{\mu_k}{2}
\left(h(x)-1\right)^2,
$$

with an increasing penalty parameter $\mu_k$.

### Method of Multipliers

The augmented Lagrangian is

$$
L_\mu(x,\lambda)
=
f(x)
+
\lambda \left(h(x)-1\right)
+
\frac{\mu}{2}
\left(h(x)-1\right)^2,
$$

followed by the multiplier update

$$
\lambda_{k+1}
=
\lambda_k
+
\mu_k
\left(h(x_k)-1\right).
$$

The implementation increases $\mu$ only when the constraint violation is not decreasing sufficiently.

### Unconstrained Subproblems

Both constrained methods call the same reusable BFGS routine from `optimizers/unconstrained.py`.

Each BFGS step uses:

- Armijo backtracking line search
- an inverse-Hessian approximation
- safeguards for small curvature values

This keeps the inner unconstrained solver independent from the outer constrained optimization algorithms.

---

## Q2 Results

Using the default initial point

```text
x0 = [0.1, 0.2, 0.7]
```

and tolerance $10^{-6}$, the refactored implementation produces:

| Method | Outer Iterations | Objective | Solution Error $\|x-x^\star\|_2$ | Final Constraint Violation |
|---|---:|---:|---:|---:|
| Quadratic penalty | 8 | 0.5454544264 | 1.62e-7 | 1.09e-7 |
| Method of multipliers | 8 | 0.5454543394 | 1.60e-7 | 1.89e-7 |

Both methods converge very close to the analytical solution.

On this run, they require the same number of outer iterations. The method of multipliers has a marginally smaller solution error, while the quadratic penalty method finishes with a slightly smaller equality-constraint violation.

The experiment also repeats the comparison from four initial points:

```text
[0.1, 0.2, 0.7]
[0.0, 0.0, 1.0]
[1.0, 0.0, 0.0]
[0.5, 0.5, 0.0]
```

For these starting points, both methods again converge in 8 outer iterations with solution errors on the order of $10^{-7}$, illustrating the robustness of the methods on this convex quadratic problem.

## Installation

Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd constrained_optimization_lab

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Dependencies are intentionally minimal:

- `numpy`
- `matplotlib`

---

## Running the Experiments

### Q1: Duality, KKT, and Sensitivity

Run the full Q1 analysis:

```bash
python scripts/run_q1_analysis.py
```

To save the figures without opening plot windows:

```bash
python scripts/run_q1_analysis.py --no-show --save-dir outputs
```

This generates:

```text
outputs/q1_value_sensitivity.png
outputs/q1_dual_functions.png
```

### Q2: Penalty vs. Method of Multipliers

Run the method comparison:

```bash
python scripts/run_q2_comparison.py
```

For iteration-by-iteration optimizer output:

```bash
python scripts/run_q2_comparison.py --verbose
```

To save the comparison figure:

```bash
python scripts/run_q2_comparison.py --no-show --save-dir outputs
```

This generates:

```text
outputs/q2_method_comparison.png
```

---

## Module Guide

### Objectives

- **`objectives/norm_constraint.py`** — primal Q1 problem definitions and analytical solutions.
- **`objectives/quadratic_eq.py`** — Q2 objective, gradient, equality constraint, and analytical KKT solution.

### Analysis

- **`analysis/duality.py`** — Q1 dual functions, optimal multipliers, and duality-gap calculations.
- **`analysis/kkt.py`** — KKT checks, including the active box-bound multiplier in Problem 2.
- **`analysis/sensitivity.py`** — optimal-value functions and finite-difference sensitivity checks.

### Optimizers

- **`optimizers/unconstrained.py`** — BFGS with Armijo backtracking for unconstrained subproblems.
- **`optimizers/base_alm.py`** — common interface and functionality for equality-constrained methods.
- **`optimizers/penalty.py`** — quadratic penalty method.
- **`optimizers/multipliers.py`** — method of multipliers / augmented Lagrangian method.

### Utilities

- **`utils/plotting.py`** — visualizations for both optimization studies.
- **`utils/comparators.py`** — Q2 method construction, performance metrics, and initial-point sensitivity experiments.

---

## Main Takeaways

- Strong duality holds for both Q1 formulations at the studied point, with zero primal-dual gap.
- Adding the box domain leaves the primal optimum unchanged at $r=1$, but makes the $\ell_1$-constraint multiplier non-unique.
- The multiplier interval in Q1 is directly connected to the kink in the optimal-value function and its one-sided sensitivities.
- Both the quadratic penalty method and method of multipliers accurately solve the Q2 equality-constrained quadratic problem.
- Separating the BFGS inner solver from the outer constrained methods makes the implementation easier to inspect, test, and extend.