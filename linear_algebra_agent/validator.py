"""validator.py — Verificação de consistência e formatação de explicações."""

from __future__ import annotations

from typing import Iterable, List

import sympy as sp

from .core import MatrixLike, to_matrix, to_vector


# ---------------------------------------------------------------------------
# Verificações de consistência matemática
# ---------------------------------------------------------------------------

def check_solution(A: MatrixLike, x: MatrixLike, b: MatrixLike, tol: float = 1e-9) -> bool:
    """Confirma se ``A x ≈ b`` (simbolicamente ou numericamente)."""
    A_m, x_v, b_v = to_matrix(A), to_vector(x), to_vector(b)
    residual = sp.simplify(A_m * x_v - b_v)
    if all(sp.simplify(v) == 0 for v in residual):
        return True
    try:
        rn = residual.evalf()
        return all(abs(float(v)) < tol for v in rn)
    except (TypeError, ValueError):
        return False


def check_inverse(A: MatrixLike, A_inv: MatrixLike, tol: float = 1e-9) -> bool:
    A_m, B = to_matrix(A), to_matrix(A_inv)
    if A_m.shape != B.shape:
        return False
    prod = sp.simplify(A_m * B)
    eye = sp.eye(A_m.shape[0])
    diff = prod - eye
    if all(sp.simplify(v) == 0 for v in diff):
        return True
    try:
        return all(abs(float(v)) < tol for v in diff.evalf())
    except (TypeError, ValueError):
        return False


def check_eigenpair(A: MatrixLike, eigenvalue: sp.Expr, eigenvector: MatrixLike) -> bool:
    A_m = to_matrix(A)
    v = to_vector(eigenvector)
    diff = sp.simplify(A_m * v - eigenvalue * v)
    return all(sp.simplify(c) == 0 for c in diff)


def check_orthogonality(vectors: Iterable[MatrixLike], tol: float = 1e-9) -> bool:
    cols = [to_vector(v) for v in vectors]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            inner = sp.simplify((cols[i].T * cols[j])[0, 0])
            if inner != 0:
                try:
                    if abs(float(inner)) > tol:
                        return False
                except (TypeError, ValueError):
                    return False
    return True


# ---------------------------------------------------------------------------
# Formatação de explicações
# ---------------------------------------------------------------------------

def format_steps(steps: List[str], title: str | None = None) -> str:
    """Formata uma lista de passos como texto estruturado."""
    if not steps:
        return "(sem passos registrados)"
    out = []
    if title:
        out.append(f"=== {title} ===")
    for i, s in enumerate(steps, 1):
        out.append(f"[{i}] {s}")
    return "\n".join(out)


def pretty_matrix(M: MatrixLike) -> str:
    return sp.pretty(to_matrix(M))
