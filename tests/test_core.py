"""Testes para linear_algebra_agent.core."""

from __future__ import annotations

import sympy as sp
import pytest

from linear_algebra_agent import core


def test_to_matrix_from_list():
    M = core.to_matrix([[1, 2], [3, 4]])
    assert M.shape == (2, 2)
    assert M[0, 1] == 2


def test_to_vector_handles_row_and_column():
    v = core.to_vector([1, 2, 3])
    assert v.shape == (3, 1)
    w = core.to_vector(sp.Matrix([[1, 2, 3]]))
    assert w.shape == (3, 1)


def test_basic_matrix_operations():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert core.add(A, B) == sp.Matrix([[6, 8], [10, 12]])
    assert core.sub(B, A) == sp.Matrix([[4, 4], [4, 4]])
    assert core.matmul(A, B) == sp.Matrix([[19, 22], [43, 50]])
    assert core.transpose(A) == sp.Matrix([[1, 3], [2, 4]])
    assert core.scalar_mul(2, A) == sp.Matrix([[2, 4], [6, 8]])


def test_determinant_inverse_and_trace():
    A = [[1, 2], [3, 4]]
    assert core.determinant(A) == -2
    assert core.trace(A) == 5
    inv = core.inverse(A)
    assert (sp.Matrix(A) * inv) == sp.eye(2)


def test_inverse_singular_raises():
    with pytest.raises(ValueError):
        core.inverse([[1, 2], [2, 4]])


def test_dot_cross_norm():
    u = [1, 0, 0]
    v = [0, 1, 0]
    assert core.dot(u, v) == 0
    assert core.cross(u, v) == sp.Matrix([0, 0, 1])
    assert core.norm([3, 4]) == 5


def test_normalize_and_angle():
    n = core.normalize([3, 4])
    assert n == sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)])
    ang = core.angle([1, 0], [0, 1])
    assert sp.simplify(ang - sp.pi / 2) == 0


def test_constructors():
    assert core.identity(3) == sp.eye(3)
    assert core.zeros(2, 3) == sp.zeros(2, 3)
    assert core.diag([1, 2, 3]) == sp.diag(1, 2, 3)
