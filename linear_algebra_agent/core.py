"""core.py — Operações fundamentais de matrizes e vetores.

Fornece uma camada uniforme sobre :mod:`sympy` (para cálculos exatos /
simbólicos) com fallback para :mod:`numpy` quando solicitado. As funções
aceitam entradas em formatos variados (lista de listas, ``np.ndarray``,
``sympy.Matrix``) e retornam ``sympy.Matrix`` por padrão.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Union

import numpy as np
import sympy as sp

MatrixLike = Union[sp.Matrix, np.ndarray, Sequence[Sequence[Any]], Sequence[Any]]


# ---------------------------------------------------------------------------
# Conversões e utilitários
# ---------------------------------------------------------------------------

def to_matrix(data: MatrixLike) -> sp.Matrix:
    """Converte qualquer entrada compatível em ``sympy.Matrix``.

    Vetores 1D viram matrizes coluna.
    """
    if isinstance(data, sp.MatrixBase):
        return sp.Matrix(data)
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return sp.Matrix(data.tolist())
        return sp.Matrix(data.tolist())
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return sp.Matrix(0, 0, [])
        first = data[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return sp.Matrix([list(row) for row in data])
        return sp.Matrix(list(data))
    raise TypeError(f"Tipo não suportado para matriz: {type(data)!r}")


def to_vector(data: MatrixLike) -> sp.Matrix:
    """Converte para vetor coluna (``n x 1``)."""
    M = to_matrix(data)
    if M.shape[1] == 1:
        return M
    if M.shape[0] == 1:
        return M.T
    raise ValueError(f"Esperado vetor, recebida matriz de forma {M.shape}.")


def to_numpy(data: MatrixLike, dtype: Any = float) -> np.ndarray:
    """Converte para ``numpy.ndarray`` de ponto flutuante."""
    M = to_matrix(data)
    return np.array(M.evalf().tolist(), dtype=dtype)


def shape(data: MatrixLike) -> tuple[int, int]:
    return to_matrix(data).shape


def is_square(data: MatrixLike) -> bool:
    r, c = shape(data)
    return r == c


# ---------------------------------------------------------------------------
# Operações básicas
# ---------------------------------------------------------------------------

def add(a: MatrixLike, b: MatrixLike) -> sp.Matrix:
    A, B = to_matrix(a), to_matrix(b)
    if A.shape != B.shape:
        raise ValueError(f"Formas incompatíveis para soma: {A.shape} vs {B.shape}.")
    return A + B


def sub(a: MatrixLike, b: MatrixLike) -> sp.Matrix:
    A, B = to_matrix(a), to_matrix(b)
    if A.shape != B.shape:
        raise ValueError(f"Formas incompatíveis para subtração: {A.shape} vs {B.shape}.")
    return A - B


def scalar_mul(scalar: Any, a: MatrixLike) -> sp.Matrix:
    return sp.sympify(scalar) * to_matrix(a)


def matmul(a: MatrixLike, b: MatrixLike) -> sp.Matrix:
    A, B = to_matrix(a), to_matrix(b)
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis para produto: {A.shape} * {B.shape}."
        )
    return A * B


def transpose(a: MatrixLike) -> sp.Matrix:
    return to_matrix(a).T


def conjugate_transpose(a: MatrixLike) -> sp.Matrix:
    return to_matrix(a).H


def trace(a: MatrixLike) -> sp.Expr:
    A = to_matrix(a)
    if not A.is_square:
        raise ValueError("Traço é definido apenas para matrizes quadradas.")
    return A.trace()


def determinant(a: MatrixLike) -> sp.Expr:
    A = to_matrix(a)
    if not A.is_square:
        raise ValueError("Determinante exige matriz quadrada.")
    return A.det()


def inverse(a: MatrixLike) -> sp.Matrix:
    A = to_matrix(a)
    if not A.is_square:
        raise ValueError("Inversa exige matriz quadrada.")
    if A.det() == 0:
        raise ValueError("Matriz singular: não é inversível.")
    return A.inv()


def pseudoinverse(a: MatrixLike) -> sp.Matrix:
    """Pseudoinversa de Moore–Penrose."""
    return to_matrix(a).pinv()


def power(a: MatrixLike, k: int) -> sp.Matrix:
    A = to_matrix(a)
    if not A.is_square:
        raise ValueError("Potência exige matriz quadrada.")
    return A ** k


# ---------------------------------------------------------------------------
# Vetores
# ---------------------------------------------------------------------------

def dot(u: MatrixLike, v: MatrixLike) -> sp.Expr:
    a = to_vector(u)
    b = to_vector(v)
    if a.shape != b.shape:
        raise ValueError("Vetores de dimensões diferentes.")
    return (a.T * b)[0, 0]


def cross(u: MatrixLike, v: MatrixLike) -> sp.Matrix:
    a, b = to_vector(u), to_vector(v)
    if a.shape != (3, 1) or b.shape != (3, 1):
        raise ValueError("Produto vetorial só é definido em R^3.")
    return a.cross(b)


def norm(u: MatrixLike, p: Any = 2) -> sp.Expr:
    return to_vector(u).norm(p)


def normalize(u: MatrixLike) -> sp.Matrix:
    v = to_vector(u)
    n = v.norm()
    if n == 0:
        raise ValueError("Vetor nulo não pode ser normalizado.")
    return v / n


def angle(u: MatrixLike, v: MatrixLike) -> sp.Expr:
    """Ângulo (em radianos) entre dois vetores."""
    a, b = to_vector(u), to_vector(v)
    cos_t = dot(a, b) / (a.norm() * b.norm())
    return sp.acos(sp.simplify(cos_t))


# ---------------------------------------------------------------------------
# Construção
# ---------------------------------------------------------------------------

def identity(n: int) -> sp.Matrix:
    return sp.eye(n)


def zeros(rows: int, cols: int | None = None) -> sp.Matrix:
    return sp.zeros(rows, cols if cols is not None else rows)


def ones(rows: int, cols: int | None = None) -> sp.Matrix:
    return sp.ones(rows, cols if cols is not None else rows)


def diag(values: Iterable[Any]) -> sp.Matrix:
    return sp.diag(*list(values))
