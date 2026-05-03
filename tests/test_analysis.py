"""Testes para linear_algebra_agent.analysis."""

from __future__ import annotations

import sympy as sp

from linear_algebra_agent import analysis


def test_rank_and_nullity():
    A = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]
    r, n = analysis.rank_nullity(A)
    assert r == 2
    assert n == 1


def test_column_space_basis():
    A = [[1, 2, 1], [2, 4, 3], [3, 6, 5]]
    basis = analysis.column_space(A)
    assert len(basis) == 2


def test_null_space_solves_homogeneous():
    A = sp.Matrix([[1, 2, 3], [2, 4, 6]])
    for v in analysis.null_space(A):
        assert sp.simplify(A * v) == sp.zeros(2, 1)


def test_linear_independence():
    assert analysis.is_linearly_independent([[1, 0], [0, 1]])
    assert not analysis.is_linearly_independent([[1, 2], [2, 4]])


def test_basis_extraction():
    vectors = [[1, 0, 0], [2, 0, 0], [0, 1, 0]]
    basis = analysis.basis_of(vectors)
    assert len(basis) == 2


def test_coordinates_in_basis_round_trip():
    e1, e2 = sp.Matrix([1, 1]), sp.Matrix([1, -1])
    v = sp.Matrix([3, 1])
    coords = analysis.coordinates_in_basis(v, [e1, e2])
    assert coords[0] * e1 + coords[1] * e2 == v


def test_change_of_basis_identity():
    e1, e2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    P = analysis.change_of_basis_matrix([e1, e2], [e1, e2])
    assert P == sp.eye(2)


def test_gram_schmidt_orthogonal():
    vs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    ortho = analysis.gram_schmidt(vs)
    for i in range(len(ortho)):
        for j in range(i + 1, len(ortho)):
            assert sp.simplify((ortho[i].T * ortho[j])[0, 0]) == 0


def test_project_vector_onto_axis():
    proj = analysis.project_vector([3, 4], [1, 0])
    assert proj == sp.Matrix([3, 0])


def test_project_onto_subspace():
    proj = analysis.project_onto_subspace(
        [1, 1, 1], [[1, 0, 0], [0, 1, 0]]
    )
    assert proj == sp.Matrix([1, 1, 0])


def test_symmetry_and_orthogonality():
    A = sp.Matrix([[1, 2], [2, 1]])
    assert analysis.is_symmetric(A)
    Q = sp.Matrix([[0, 1], [1, 0]])
    assert analysis.is_orthogonal(Q)


def test_characteristic_polynomial():
    poly = analysis.characteristic_polynomial([[2, 0], [0, 3]])
    lam = list(poly.free_symbols)[0]
    assert sp.expand(poly) == sp.expand((lam - 2) * (lam - 3))


def test_diagonalizable():
    assert analysis.is_diagonalizable([[2, 0], [0, 3]])
    assert not analysis.is_diagonalizable([[1, 1], [0, 1]])
