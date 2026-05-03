"""Testes integrados para LinearAlgebraAgent."""

from __future__ import annotations

import sympy as sp

from linear_algebra_agent import LinearAlgebraAgent


def test_agent_solves_system_from_text():
    agent = LinearAlgebraAgent(explain=True)
    res = agent.solve("Resolver sistema [[2,1],[1,3]] com b = [5, 10]")
    assert res.operation == "solve_system"
    assert res.value == sp.Matrix([1, 3])
    assert res.valid is True
    assert res.explanation


def test_agent_determinant_from_text():
    agent = LinearAlgebraAgent()
    res = agent.solve("Calcule o determinante de [[1,2],[3,4]]")
    assert res.value == -2


def test_agent_inverse_validates():
    agent = LinearAlgebraAgent()
    res = agent.solve("Encontre a matriz inversa de [[1,2],[3,4]]")
    assert res.valid is True
    assert res.value == sp.Matrix([[1, 2], [3, 4]]).inv()


def test_agent_eigen():
    agent = LinearAlgebraAgent()
    res = agent.solve("Calcule autovalores e autovetores de [[2,0],[0,3]]")
    assert res.operation == "eigen"
    assert res.valid is True


def test_agent_gram_schmidt():
    agent = LinearAlgebraAgent()
    res = agent.solve("Aplique gram-schmidt aos vetores [1,1,0], [1,0,1], [0,1,1]")
    assert res.operation == "gram_schmidt"
    assert res.valid is True


def test_agent_projection():
    agent = LinearAlgebraAgent()
    res = agent.solve("Projeção do vetor [3,4] sobre [1,0]")
    assert res.value == sp.Matrix([3, 0])


def test_agent_register_custom_handler():
    agent = LinearAlgebraAgent()

    def handler(parsed):
        from linear_algebra_agent.agent import AgentResult
        return AgentResult("doubled", value=2 * parsed.matrices[0])

    from linear_algebra_agent.interface import register_operation
    register_operation("doubled", ["dobrar matriz"])
    agent.register_handler("doubled", handler)
    res = agent.solve("Dobrar matriz [[1,2],[3,4]]")
    assert res.value == sp.Matrix([[2, 4], [6, 8]])


def test_pretty_output_runs():
    agent = LinearAlgebraAgent(explain=True)
    res = agent.solve("Resolver sistema [[1,0],[0,1]] com b = [1, 2]")
    txt = res.pretty()
    assert "Operação" in txt
