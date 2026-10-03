"""Small unconstrained optimizer used for ALM subproblems."""

from typing import Callable

import numpy as np

from utils.dataclasses import UnconstrainedResult

ArrayFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]


def armijo_backtracking(
    func: ArrayFunction,
    grad_func: GradientFunction,
    x: np.ndarray,
    direction: np.ndarray,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    min_alpha: float = 1e-12,
) -> float:
    """Backtracking line search satisfying the Armijo decrease condition."""
    alpha = float(alpha0)
    fx = func(x)
    grad = grad_func(x)
    directional_derivative = float(grad @ direction)

    while func(x + alpha * direction) > fx + c1 * alpha * directional_derivative:
        alpha *= rho
        if alpha < min_alpha:
            return min_alpha
    return alpha


def bfgs_minimize(
    func: ArrayFunction,
    grad_func: GradientFunction,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> UnconstrainedResult:
    """Minimize a differentiable function with an inverse-Hessian BFGS update."""
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    inverse_hessian = np.eye(n)
    grad = grad_func(x)

    if np.linalg.norm(grad) < tol:
        return UnconstrainedResult(x, func(x), 0, float(np.linalg.norm(grad)), True)

    for iteration in range(1, max_iter + 1):
        direction = -inverse_hessian @ grad
        if float(grad @ direction) >= 0.0:
            direction = -grad
            inverse_hessian = np.eye(n)

        alpha = armijo_backtracking(func, grad_func, x, direction)
        x_new = x + alpha * direction
        grad_new = grad_func(x_new)
        grad_norm = float(np.linalg.norm(grad_new))

        if grad_norm < tol:
            return UnconstrainedResult(x_new, func(x_new), iteration, grad_norm, True)

        s = x_new - x
        y = grad_new - grad
        curvature = float(y @ s)

        if curvature > 1e-12:
            rho = 1.0 / curvature
            identity = np.eye(n)
            left = identity - rho * np.outer(s, y)
            right = identity - rho * np.outer(y, s)
            inverse_hessian = left @ inverse_hessian @ right + rho * np.outer(s, s)
        else:
            inverse_hessian = np.eye(n)

        x = x_new
        grad = grad_new

    grad_norm = float(np.linalg.norm(grad))
    return UnconstrainedResult(x, func(x), max_iter, grad_norm, grad_norm < tol)
