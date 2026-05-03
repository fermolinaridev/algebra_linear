"""Exemplo 2 — Autovalores, autovetores e diagonalização."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sympy as sp

from linear_algebra_agent import LinearAlgebraAgent, solvers, analysis


def main() -> None:
    A = sp.Matrix([[4, 1], [2, 3]])
    print("A =")
    print(sp.pretty(A))

    print("\n== Autovalores e autovetores ==")
    for lam, mult, vecs in solvers.eigen(A):
        print(f"λ = {lam} (multiplicidade {mult})")
        for v in vecs:
            print("  autovetor =", list(v))

    print("\n== Polinômio característico ==")
    print(analysis.characteristic_polynomial(A))

    print("\n== Diagonalização A = P D P^-1 ==")
    P, D = solvers.spectral_decomposition(A)
    print("P =")
    print(sp.pretty(P))
    print("D =")
    print(sp.pretty(D))
    print("Verificação P D P^-1 - A:", sp.simplify(P * D * P.inv() - A))

    print("\n== Via agente em linguagem natural ==")
    agent = LinearAlgebraAgent()
    res = agent.solve("Calcule autovalores e autovetores de [[4,1],[2,3]]")
    print(res.pretty())


if __name__ == "__main__":
    main()
