"""analysis.py — Propriedades algébricas e espaços vetoriais.

Funções para posto, nulidade, independência linear, bases, mudança de base,
projeções e ortogonalização (Gram–Schmidt).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import sympy as sp

from .core import MatrixLike, to_matrix, to_vector


# ---------------------------------------------------------------------------
# Postos e dimensões
# ---------------------------------------------------------------------------

def rank(A: MatrixLike) -> int:
    return to_matrix(A).rank()


def nullity(A: MatrixLike) -> int:
    M = to_matrix(A)
    return M.shape[1] - M.rank()


def rank_nullity(A: MatrixLike) -> Tuple[int, int]:
    M = to_matrix(A)
    r = M.rank()
    return r, M.shape[1] - r


# ---------------------------------------------------------------------------
# Espaços fundamentais
# ---------------------------------------------------------------------------

def column_space(A: MatrixLike) -> List[sp.Matrix]:
    """Base do espaço-coluna."""
    return to_matrix(A).columnspace()


def row_space(A: MatrixLike) -> List[sp.Matrix]:
    """Base do espaço-linha (vetores linha como :class:`Matrix` 1×n)."""
    return to_matrix(A).rowspace()


def null_space(A: MatrixLike) -> List[sp.Matrix]:
    """Base do núcleo (kernel) de A."""
    return to_matrix(A).nullspace()


def left_null_space(A: MatrixLike) -> List[sp.Matrix]:
    """Base do núcleo à esquerda — núcleo de A^T."""
    return to_matrix(A).T.nullspace()


# ---------------------------------------------------------------------------
# Independência linear, bases e mudanças de base
# ---------------------------------------------------------------------------

def is_linearly_independent(vectors: Sequence[MatrixLike]) -> bool:
    if not vectors:
        return True
    cols = [to_vector(v) for v in vectors]
    M = sp.Matrix.hstack(*cols)
    return M.rank() == len(vectors)


def basis_of(vectors: Sequence[MatrixLike]) -> List[sp.Matrix]:
    """Extrai uma base do subespaço gerado por ``vectors``."""
    if not vectors:
        return []
    cols = [to_vector(v) for v in vectors]
    M = sp.Matrix.hstack(*cols)
    _, pivots = M.rref()
    return [cols[j] for j in pivots]


def coordinates_in_basis(
    v: MatrixLike, basis: Sequence[MatrixLike]
) -> sp.Matrix:
    """Coordenadas de ``v`` na base ``basis``."""
    if not basis:
        raise ValueError("Base vazia.")
    cols = [to_vector(b) for b in basis]
    B = sp.Matrix.hstack(*cols)
    target = to_vector(v)
    sol = B.solve(target)
    return sol


def change_of_basis_matrix(
    from_basis: Sequence[MatrixLike], to_basis: Sequence[MatrixLike]
) -> sp.Matrix:
    """Matriz que converte coordenadas de ``from_basis`` para ``to_basis``.

    ``[v]_to = P · [v]_from``.
    """
    F = sp.Matrix.hstack(*[to_vector(v) for v in from_basis])
    T = sp.Matrix.hstack(*[to_vector(v) for v in to_basis])
    if T.shape != F.shape:
        raise ValueError("Bases com dimensões diferentes.")
    return T.inv() * F


# ---------------------------------------------------------------------------
# Ortogonalização e projeções
# ---------------------------------------------------------------------------

def gram_schmidt(
    vectors: Sequence[MatrixLike], normalize: bool = False
) -> List[sp.Matrix]:
    """Aplica Gram–Schmidt a ``vectors`` (descarta vetores dependentes)."""
    cols = [to_vector(v) for v in vectors]
    return sp.GramSchmidt(cols, orthonormal=normalize)


def project_vector(v: MatrixLike, onto: MatrixLike) -> sp.Matrix:
    """Projeção ortogonal de ``v`` sobre o vetor ``onto``."""
    a = to_vector(v)
    u = to_vector(onto)
    denom = (u.T * u)[0, 0]
    if denom == 0:
        raise ValueError("Não é possível projetar sobre o vetor nulo.")
    coef = (u.T * a)[0, 0] / denom
    return sp.simplify(coef) * u


def project_onto_subspace(
    v: MatrixLike, basis: Sequence[MatrixLike]
) -> sp.Matrix:
    """Projeção ortogonal de ``v`` sobre o subespaço gerado por ``basis``."""
    a = to_vector(v)
    ortho = gram_schmidt(basis, normalize=False)
    proj = sp.zeros(a.shape[0], 1)
    for u in ortho:
        denom = (u.T * u)[0, 0]
        coef = (u.T * a)[0, 0] / denom
        proj += coef * u
    return sp.simplify(proj)


# ---------------------------------------------------------------------------
# Propriedades de matrizes
# ---------------------------------------------------------------------------

def is_symmetric(A: MatrixLike) -> bool:
    M = to_matrix(A)
    return M.is_square and M == M.T


def is_orthogonal(A: MatrixLike) -> bool:
    M = to_matrix(A)
    if not M.is_square:
        return False
    return sp.simplify(M.T * M - sp.eye(M.shape[0])) == sp.zeros(*M.shape)


def is_positive_definite(A: MatrixLike) -> bool:
    M = to_matrix(A)
    if not is_symmetric(M):
        return False
    try:
        M.cholesky()
        return True
    except Exception:
        return False


def characteristic_polynomial(A: MatrixLike, var: str = "λ") -> sp.Expr:
    M = to_matrix(A)
    if not M.is_square:
        raise ValueError("Polinômio característico exige matriz quadrada.")
    return M.charpoly(sp.Symbol(var)).as_expr()


def minimal_polynomial(A: MatrixLike, var: str = "λ") -> sp.Expr:
    return sp.minimal_polynomial(to_matrix(A), sp.Symbol(var))


def is_diagonalizable(A: MatrixLike) -> bool:
    M = to_matrix(A)
    if not M.is_square:
        return False
    return M.is_diagonalizable()
