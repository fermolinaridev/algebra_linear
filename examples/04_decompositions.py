"""Exemplo 4 — Decomposições LU, QR e SVD."""

from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sympy as sp

from linear_algebra_agent import solvers


def main() -> None:
    A = sp.Matrix([[4, 3], [6, 3]])

    print("== LU ==")
    L, U, perm = solvers.lu_decomposition(A)
    print("L =\n" + sp.pretty(L))
    print("U =\n" + sp.pretty(U))
    print("Permutações:", perm)

    print("\n== QR (Gram–Schmidt simbólico) ==")
    Q, R = solvers.qr_decomposition(sp.Matrix([[1, 1], [1, 0], [0, 1]]))
    print("Q =\n" + sp.pretty(Q))
    print("R =\n" + sp.pretty(R))

    print("\n== SVD (numérica) ==")
    U, s, Vt = solvers.svd([[1, 0, 1], [0, 1, 1]])
    print("Valores singulares:", s)
    print("U.shape:", U.shape, " Vt.shape:", Vt.shape)


if __name__ == "__main__":
    main()
