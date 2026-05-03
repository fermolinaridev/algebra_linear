"""Testes para linear_algebra_agent.interface."""

from __future__ import annotations

import sympy as sp

from linear_algebra_agent import interface


def test_extract_matrix_and_vector():
    text = "Calcule det de [[1,2],[3,4]] e use o vetor [5, 6]."
    mats, vecs, _ = interface.extract_arrays(text)
    assert len(mats) == 1
    assert len(vecs) == 1
    assert mats[0].shape == (2, 2)
    assert vecs[0].shape == (2, 1)


def test_detect_operations():
    assert interface.detect_operation("Resolver sistema A x = b") == "solve_system"
    assert interface.detect_operation("Calcule o determinante") == "determinant"
    assert interface.detect_operation("Calcule autovalores e autovetores") == "eigen"
    assert interface.detect_operation("Decomposição QR da matriz") == "qr"
    assert interface.detect_operation("Aplique Gram-Schmidt aos vetores") == "gram_schmidt"


def test_parse_problem_solve_system():
    text = "Resolva o sistema [[2,1],[1,3]] e b = [5, 10]"
    parsed = interface.parse_problem(text)
    assert parsed.operation == "solve_system"
    assert parsed.matrices and parsed.vectors
    assert "A" in parsed.extras
    assert "b" in parsed.extras


def test_parse_problem_determinant():
    parsed = interface.parse_problem("Calcule o determinante de [[1,2],[3,4]]")
    assert parsed.operation == "determinant"
    assert parsed.matrices[0] == sp.Matrix([[1, 2], [3, 4]])


def test_register_operation_extension():
    interface.register_operation("custom_op", ["operacao customizada"])
    parsed = interface.parse_problem("aplique operacao customizada em [[1,0],[0,1]]")
    assert parsed.operation == "custom_op"
    # cleanup
    interface.OPERATION_KEYWORDS.pop("custom_op", None)
