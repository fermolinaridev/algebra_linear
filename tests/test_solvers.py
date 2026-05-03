"""Testes para linear_algebra_agent.solvers."""

from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from linear_algebra_agent import solvers


def test_solve_unique():
    A = [[2, 1], [1, 3]]
    b = [5, 10]
    res = solvers.solve_system(A, b, explain=True)
    assert res.kind == "unique"
    assert res.solution == sp.Matrix([1, 3])
    assert res.steps  # explicações geradas


def test_solve_inconsistent():
    A = [[1, 2], [2, 4]]
    b = [3, 7]
    res = solvers.solve_system(A, b)
    assert res.kind == "inconsistent"


def test_solve_infinite_solutions():
    A = [[1, 2, 3], [2, 4, 6]]
    b = [6, 12]
    res = solvers.solve_system(A, b)
    assert res.kind == "infinite"
    assert len(res.free_symbols) == 2


def test_cramer_method():
    A = [[3, 2], [1, 2]]
    b = [16, 8]
    res = solvers.solve_system(A, b, method="cramer")
    assert res.kind == "unique"
    assert res.solution == sp.Matrix([4, 2])


def test_lu_method():
    A = [[4, 3], [6, 3]]
    b = [10, 12]
    res = solvers.solve_system(A, b, method="lu")
    assert res.kind == "unique"
    assert sp.simplify(sp.Matrix(A) * res.solution - sp.Matrix([b]).T) == sp.zeros(2, 1)


def test_jacobi_iteration_converges():
    A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
    b = [6, 25, -11]
    x, _ = solvers.jacobi(A, b, tol=1e-9, max_iter=200)
    assert np.allclose(np.array(A, dtype=float) @ x, b, atol=1e-6)


def test_gauss_seidel_converges():
    A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
    b = [6, 25, -11]
    x, _ = solvers.gauss_seidel(A, b, tol=1e-10, max_iter=200)
    assert np.allclose(np.array(A, dtype=float) @ x, b, atol=1e-7)


def test_lu_decomposition_recovers_a():
    A = sp.Matrix([[4, 3], [6, 3]])
    L, U, perm = solvers.lu_decomposition(A)
    P = sp.eye(2)
    for i, j in perm:
        P.row_swap(i, j)
    assert sp.simplify(P * A - L * U) == sp.zeros(2, 2)


def test_qr_decomposition_orthogonal():
    A = sp.Matrix([[1, 1], [1, 0], [0, 1]])
    Q, R = solvers.qr_decomposition(A)
    assert sp.simplify(Q.T * Q - sp.eye(Q.shape[1])) == sp.zeros(Q.shape[1], Q.shape[1])
    assert sp.simplify(Q * R - A) == sp.zeros(*A.shape)


def test_eigen_simple():
    A = [[2, 0], [0, 3]]
    spectrum = solvers.eigen(A)
    eigenvalues = sorted(int(lam) for lam, _, _ in spectrum)
    assert eigenvalues == [2, 3]


def test_spectral_decomposition():
    A = sp.Matrix([[2, 1], [1, 2]])
    P, D = solvers.spectral_decomposition(A)
    assert sp.simplify(P * D * P.inv() - A) == sp.zeros(2, 2)


def test_least_squares():
    A = [[1, 1], [1, 2], [1, 3]]
    b = [1, 2, 2]
    x = solvers.least_squares(A, b)
    expected = sp.Matrix([sp.Rational(2, 3), sp.Rational(1, 2)])
    assert sp.simplify(x - expected) == sp.zeros(2, 1)


def test_svd_returns_correct_shapes():
    U, s, Vt = solvers.svd([[1, 0], [0, 2]])
    assert U.shape == (2, 2)
    assert Vt.shape == (2, 2)
    assert sorted(s.tolist()) == [1.0, 2.0]
