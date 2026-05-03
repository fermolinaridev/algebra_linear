"""Exemplo 3 — Bases, mudanças de base e projeções ortogonais."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sympy as sp

from linear_algebra_agent import LinearAlgebraAgent, analysis


def main() -> None:
    print("== Verificando independência linear ==")
    vs = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
    print("Vetores são independentes?", analysis.is_linearly_independent(vs))

    print("\n== Extraindo base do subespaço gerado ==")
    base = analysis.basis_of(vs)
    for b in base:
        print("  base =", list(b))

    print("\n== Coordenadas em outra base ==")
    e1 = sp.Matrix([1, 1])
    e2 = sp.Matrix([1, -1])
    coords = analysis.coordinates_in_basis(sp.Matrix([3, 1]), [e1, e2])
    print("[v]_B =", list(coords))

    print("\n== Mudança de base ==")
    P = analysis.change_of_basis_matrix(
        from_basis=[sp.Matrix([1, 0]), sp.Matrix([1, 1])],
        to_basis=[sp.Matrix([1, 0]), sp.Matrix([0, 1])],
    )
    print("P =\n" + sp.pretty(P))

    print("\n== Gram–Schmidt ==")
    ortho = analysis.gram_schmidt([[1, 1, 0], [1, 0, 1], [0, 1, 1]], normalize=True)
    for u in ortho:
        print("  u =", list(u))

    print("\n== Projeção ortogonal sobre subespaço ==")
    proj = analysis.project_onto_subspace([1, 1, 1], [[1, 0, 0], [0, 1, 0]])
    print("proj =", list(proj))

    print("\n== Via agente em linguagem natural ==")
    agent = LinearAlgebraAgent()
    print(agent.solve("Projeção do vetor [3,4] sobre [1,0]").pretty())


if __name__ == "__main__":
    main()
