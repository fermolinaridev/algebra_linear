"""Testes para linear_algebra_agent.validator."""

from __future__ import annotations

import sympy as sp

from linear_algebra_agent import validator


def test_check_solution_true():
    A = sp.Matrix([[1, 2], [3, 4]])
    x = sp.Matrix([1, 1])
    b = A * x
    assert validator.check_solution(A, x, b)


def test_check_solution_false():
    A = sp.Matrix([[1, 0], [0, 1]])
    x = sp.Matrix([1, 1])
    b = sp.Matrix([2, 2])
    assert not validator.check_solution(A, x, b)


def test_check_inverse():
    A = sp.Matrix([[1, 2], [3, 4]])
    assert validator.check_inverse(A, A.inv())
    assert not validator.check_inverse(A, sp.eye(2))


def test_check_eigenpair():
    A = sp.Matrix([[2, 0], [0, 3]])
    assert validator.check_eigenpair(A, 2, sp.Matrix([1, 0]))
    assert not validator.check_eigenpair(A, 5, sp.Matrix([1, 0]))


def test_check_orthogonality():
    assert validator.check_orthogonality([[1, 0], [0, 1]])
    assert not validator.check_orthogonality([[1, 0], [1, 1]])


def test_format_steps_and_pretty():
    txt = validator.format_steps(["primeiro", "segundo"], title="Demo")
    assert "[1] primeiro" in txt and "Demo" in txt
    assert validator.pretty_matrix([[1, 2], [3, 4]])
