"""solvers.py — Resolução de sistemas lineares e decomposições matriciais.

Inclui métodos diretos (eliminação de Gauss, LU), iterativos (Jacobi,
Gauss–Seidel) e decomposições clássicas (LU, QR, SVD, espectral).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import sympy as sp

from . import core
from .core import MatrixLike, to_matrix, to_numpy, to_vector


# ---------------------------------------------------------------------------
# Resultados estruturados
# ---------------------------------------------------------------------------

@dataclass
class LinearSystemResult:
    """Resultado de :func:`solve_system`."""

    kind: str  # "unique", "infinite", "inconsistent"
    solution: Optional[sp.Matrix] = None
    parametric: Optional[sp.Matrix] = None
    free_symbols: List[sp.Symbol] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        if self.kind == "unique":
            return f"LinearSystemResult(unique, x={list(self.solution)})"
        if self.kind == "infinite":
            return (
                f"LinearSystemResult(infinite, free={self.free_symbols}, "
                f"x={list(self.parametric)})"
            )
        return "LinearSystemResult(inconsistent)"


# ---------------------------------------------------------------------------
# Resolução de sistemas Ax = b
# ---------------------------------------------------------------------------

def solve_system(
    A: MatrixLike,
    b: MatrixLike,
    method: str = "auto",
    explain: bool = False,
) -> LinearSystemResult:
    """Resolve ``A x = b`` simbolicamente.

    Parameters
    ----------
    method:
        ``"auto"`` (padrão), ``"gauss"``, ``"lu"``, ``"cramer"``.
    explain:
        Quando verdadeiro, registra explicações passo a passo no resultado.
    """
    A_m = to_matrix(A)
    b_v = to_vector(b)
    if A_m.shape[0] != b_v.shape[0]:
        raise ValueError("Linhas de A e b devem coincidir.")

    steps: List[str] = []
    if explain:
        steps.append(f"Sistema com A de forma {A_m.shape} e b de tamanho {b_v.shape[0]}.")

    if method == "cramer":
        return _solve_cramer(A_m, b_v, steps, explain)
    if method == "lu":
        return _solve_lu(A_m, b_v, steps, explain)
    return _solve_gauss(A_m, b_v, steps, explain)


def _solve_gauss(A: sp.Matrix, b: sp.Matrix, steps: List[str], explain: bool) -> LinearSystemResult:
    aug = A.row_join(b)
    rref, pivots = aug.rref()
    if explain:
        steps.append("Construída matriz aumentada [A|b]; aplicada eliminação de Gauss-Jordan.")
        steps.append(f"Forma escalonada reduzida:\n{sp.pretty(rref)}")
        steps.append(f"Pivôs em colunas: {pivots}")

    n_cols = A.shape[1]
    # Inconsistência: pivô na última coluna
    if n_cols in pivots:
        if explain:
            steps.append("Pivô na coluna de b ⇒ sistema inconsistente.")
        return LinearSystemResult(kind="inconsistent", steps=steps)

    free = [j for j in range(n_cols) if j not in pivots]
    if not free:
        sol = rref[:n_cols, -1]
        if explain:
            steps.append("Sem variáveis livres ⇒ solução única.")
        return LinearSystemResult(kind="unique", solution=sol, steps=steps)

    syms = sp.symbols(f"t1:{len(free) + 1}")
    x = sp.zeros(n_cols, 1)
    for k, j in enumerate(free):
        x[j, 0] = syms[k]
    for row, j_pivot in enumerate(pivots):
        expr = rref[row, -1]
        for k, j_free in enumerate(free):
            expr -= rref[row, j_free] * syms[k]
        x[j_pivot, 0] = sp.simplify(expr)
    if explain:
        steps.append(f"Variáveis livres: {[f'x{j+1}' for j in free]} ⇒ solução paramétrica.")
    return LinearSystemResult(
        kind="infinite", parametric=x, free_symbols=list(syms), steps=steps
    )


def _solve_cramer(A: sp.Matrix, b: sp.Matrix, steps: List[str], explain: bool) -> LinearSystemResult:
    if not A.is_square:
        raise ValueError("Cramer exige matriz quadrada.")
    det = A.det()
    if det == 0:
        if explain:
            steps.append("det(A) = 0 ⇒ Cramer não se aplica; delegando ao método de Gauss.")
        return _solve_gauss(A, b, steps, explain)
    n = A.shape[0]
    x = sp.zeros(n, 1)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i, 0] = sp.simplify(Ai.det() / det)
        if explain:
            steps.append(f"x{i+1} = det(A_{i+1}) / det(A) = {x[i, 0]}")
    return LinearSystemResult(kind="unique", solution=x, steps=steps)


def _solve_lu(A: sp.Matrix, b: sp.Matrix, steps: List[str], explain: bool) -> LinearSystemResult:
    if not A.is_square:
        return _solve_gauss(A, b, steps, explain)
    try:
        L, U, perm = A.LUdecomposition()
    except Exception as exc:  # pragma: no cover
        if explain:
            steps.append(f"Falha em LU ({exc}); usando Gauss-Jordan.")
        return _solve_gauss(A, b, steps, explain)
    bp = b.copy()
    for i, j in perm:
        bp.row_swap(i, j)
    if explain:
        steps.append("LU obtida; resolvendo Ly = Pb e Ux = y.")
    y = L.solve(bp)
    x = U.solve(y)
    return LinearSystemResult(kind="unique", solution=x, steps=steps)


# ---------------------------------------------------------------------------
# Métodos iterativos (Jacobi e Gauss–Seidel)
# ---------------------------------------------------------------------------

def jacobi(
    A: MatrixLike,
    b: MatrixLike,
    x0: Optional[MatrixLike] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int]:
    A_n = to_numpy(A)
    b_n = to_numpy(b).reshape(-1)
    n = A_n.shape[0]
    x = np.zeros(n) if x0 is None else to_numpy(x0).reshape(-1).copy()
    D = np.diag(A_n)
    if np.any(D == 0):
        raise ValueError("Diagonal nula impede método de Jacobi.")
    R = A_n - np.diagflat(D)
    for k in range(1, max_iter + 1):
        x_new = (b_n - R @ x) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def gauss_seidel(
    A: MatrixLike,
    b: MatrixLike,
    x0: Optional[MatrixLike] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int]:
    A_n = to_numpy(A)
    b_n = to_numpy(b).reshape(-1)
    n = A_n.shape[0]
    x = np.zeros(n) if x0 is None else to_numpy(x0).reshape(-1).copy()
    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            s1 = A_n[i, :i] @ x_new[:i]
            s2 = A_n[i, i + 1 :] @ x[i + 1 :]
            if A_n[i, i] == 0:
                raise ValueError("Diagonal nula impede Gauss–Seidel.")
            x_new[i] = (b_n[i] - s1 - s2) / A_n[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


# ---------------------------------------------------------------------------
# Decomposições
# ---------------------------------------------------------------------------

def lu_decomposition(A: MatrixLike) -> Tuple[sp.Matrix, sp.Matrix, list]:
    """Retorna ``(L, U, perm)`` onde ``perm`` é a lista de trocas de linha."""
    return to_matrix(A).LUdecomposition()


def qr_decomposition(A: MatrixLike) -> Tuple[sp.Matrix, sp.Matrix]:
    """Decomposição QR (via Gram–Schmidt simbólico)."""
    return to_matrix(A).QRdecomposition()


def cholesky(A: MatrixLike) -> sp.Matrix:
    M = to_matrix(A)
    if not M.is_square:
        raise ValueError("Cholesky exige matriz quadrada simétrica positiva-definida.")
    return M.cholesky()


def svd(A: MatrixLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD numérica: retorna ``U, s, Vt`` (vetor de valores singulares)."""
    return np.linalg.svd(to_numpy(A), full_matrices=False)


def eigen(A: MatrixLike) -> List[Tuple[sp.Expr, int, List[sp.Matrix]]]:
    """Decomposição espectral simbólica.

    Retorna lista de tuplas ``(autovalor, multiplicidade, [autovetores])``.
    """
    M = to_matrix(A)
    if not M.is_square:
        raise ValueError("Autovalores exigem matriz quadrada.")
    return M.eigenvects()


def spectral_decomposition(A: MatrixLike) -> Tuple[sp.Matrix, sp.Matrix]:
    """Diagonalização ``A = P D P^{-1}`` quando possível."""
    M = to_matrix(A)
    if not M.is_square:
        raise ValueError("Diagonalização exige matriz quadrada.")
    P, D = M.diagonalize()
    return P, D


# ---------------------------------------------------------------------------
# Mínimos quadrados
# ---------------------------------------------------------------------------

def least_squares(A: MatrixLike, b: MatrixLike) -> sp.Matrix:
    """Resolve ``min ||Ax − b||_2`` via equações normais."""
    A_m = to_matrix(A)
    b_v = to_vector(b)
    AtA = A_m.T * A_m
    Atb = A_m.T * b_v
    return AtA.solve(Atb)
