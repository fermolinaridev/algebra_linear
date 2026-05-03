"""Exemplo 1 — Resolução de sistemas lineares.

Mostra os três caminhos:

* via :class:`LinearAlgebraAgent` com texto livre;
* chamando diretamente :func:`solvers.solve_system`;
* usando o método iterativo de Gauss–Seidel.
"""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import sympy as sp

from linear_algebra_agent import LinearAlgebraAgent, solvers


def main() -> None:
    print("== Sistema linear via texto ==")
    agent = LinearAlgebraAgent(explain=True)
    res = agent.solve(
        "Resolver sistema [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]] com b = [8, -11, -3]"
    )
    print(res.pretty())

    print("\n== Sistema indeterminado ==")
    res = solvers.solve_system(
        sp.Matrix([[1, 2, 3], [2, 4, 6]]), sp.Matrix([6, 12]), explain=True
    )
    print("Tipo:", res.kind)
    print("Solução paramétrica:", list(res.parametric))
    print("Símbolos livres:", res.free_symbols)

    print("\n== Iterativo (Gauss–Seidel) ==")
    A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
    b = [6, 25, -11]
    x, it = solvers.gauss_seidel(A, b, tol=1e-10)
    print(f"Convergiu em {it} iterações")
    print("x ≈", np.round(x, 6))


if __name__ == "__main__":
    main()
