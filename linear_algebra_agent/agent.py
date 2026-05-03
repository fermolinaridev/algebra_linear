"""agent.py — Classe principal :class:`LinearAlgebraAgent`.

Concentra a orquestração entre :mod:`interface` (parsing de linguagem
natural), :mod:`core` / :mod:`solvers` / :mod:`analysis` (motor matemático)
e :mod:`validator` (consistência e explicações).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import sympy as sp

from . import analysis, core, solvers, validator
from .interface import ParsedProblem, parse_problem


# ---------------------------------------------------------------------------
# Resultado padronizado
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Saída padronizada do agente."""

    operation: str
    value: Any = None
    explanation: List[str] = field(default_factory=list)
    valid: Optional[bool] = None
    parsed: Optional[ParsedProblem] = None

    def pretty(self) -> str:
        lines = [f"Operação: {self.operation}"]
        if isinstance(self.value, sp.MatrixBase):
            lines.append("Resultado:\n" + sp.pretty(self.value))
        elif isinstance(self.value, (list, tuple)):
            lines.append("Resultado:")
            for item in self.value:
                lines.append("  " + (sp.pretty(item) if isinstance(item, sp.Basic) else repr(item)))
        else:
            lines.append(f"Resultado: {self.value}")
        if self.valid is not None:
            lines.append(f"Validação: {'OK' if self.valid else 'falhou'}")
        if self.explanation:
            lines.append("Explicação:")
            for i, step in enumerate(self.explanation, 1):
                lines.append(f"  [{i}] {step}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AgentResult(operation={self.operation!r}, valid={self.valid})"


# ---------------------------------------------------------------------------
# Agente principal
# ---------------------------------------------------------------------------

class LinearAlgebraAgent:
    """Agente especializado em problemas de Álgebra Linear.

    Parameters
    ----------
    explain:
        Quando ``True``, todos os métodos registram explicação passo a passo.
    validate:
        Quando ``True``, o agente verifica a consistência de cada cálculo.
    """

    def __init__(self, explain: bool = False, validate: bool = True) -> None:
        self.explain = explain
        self.validate = validate
        self._handlers: Dict[str, Callable[[ParsedProblem], AgentResult]] = {
            "solve_system": self._h_solve_system,
            "determinant": self._h_determinant,
            "inverse": self._h_inverse,
            "transpose": self._h_transpose,
            "rank": self._h_rank,
            "nullity": self._h_nullity,
            "trace": self._h_trace,
            "eigen": self._h_eigen,
            "diagonalize": self._h_diagonalize,
            "lu": self._h_lu,
            "qr": self._h_qr,
            "svd": self._h_svd,
            "cholesky": self._h_cholesky,
            "gram_schmidt": self._h_gram_schmidt,
            "project": self._h_project,
            "rref": self._h_rref,
            "char_poly": self._h_char_poly,
            "is_independent": self._h_is_independent,
            "basis": self._h_basis,
            "least_squares": self._h_least_squares,
            "matmul": self._h_matmul,
            "add": self._h_add,
        }

    # ------------------------------------------------------------------
    # Porta de entrada principal
    # ------------------------------------------------------------------

    def solve(self, problem: str) -> AgentResult:
        """Interpreta um enunciado em texto e devolve um :class:`AgentResult`."""
        parsed = parse_problem(problem)
        handler = self._handlers.get(parsed.operation)
        if handler is None:
            raise ValueError(
                f"Operação '{parsed.operation}' não reconhecida. "
                f"Use register_operation() para estender o vocabulário."
            )
        result = handler(parsed)
        result.parsed = parsed
        return result

    # ------------------------------------------------------------------
    # Extensibilidade
    # ------------------------------------------------------------------

    def register_handler(
        self, operation: str, handler: Callable[[ParsedProblem], AgentResult]
    ) -> None:
        """Permite registrar handlers adicionais em tempo de execução."""
        self._handlers[operation] = handler

    # ------------------------------------------------------------------
    # API direta — útil quando o usuário já tem matrizes em mãos
    # ------------------------------------------------------------------

    def solve_linear_system(self, A, b, method: str = "auto") -> AgentResult:
        res = solvers.solve_system(A, b, method=method, explain=self.explain)
        if res.kind == "unique":
            value = res.solution
            valid = validator.check_solution(A, value, b) if self.validate else None
        elif res.kind == "infinite":
            value = res.parametric
            valid = None
        else:
            value = None
            valid = None
        return AgentResult("solve_system", value=value, explanation=res.steps, valid=valid)

    def determinant(self, A) -> AgentResult:
        return AgentResult("determinant", value=core.determinant(A))

    def inverse(self, A) -> AgentResult:
        inv = core.inverse(A)
        valid = validator.check_inverse(A, inv) if self.validate else None
        return AgentResult("inverse", value=inv, valid=valid)

    def eigen(self, A) -> AgentResult:
        spectrum = solvers.eigen(A)
        valid = None
        if self.validate:
            valid = all(
                validator.check_eigenpair(A, lam, v) for lam, _, vs in spectrum for v in vs
            )
        return AgentResult("eigen", value=spectrum, valid=valid)

    # ------------------------------------------------------------------
    # Handlers internos (cada um lida com um ParsedProblem)
    # ------------------------------------------------------------------

    @staticmethod
    def _require(parsed: ParsedProblem, *, mats: int = 0, vecs: int = 0) -> None:
        if len(parsed.matrices) < mats:
            raise ValueError(
                f"Operação '{parsed.operation}' exige {mats} matriz(es); "
                f"recebidas {len(parsed.matrices)}."
            )
        if len(parsed.vectors) < vecs:
            raise ValueError(
                f"Operação '{parsed.operation}' exige {vecs} vetor(es); "
                f"recebidos {len(parsed.vectors)}."
            )

    def _first_matrix(self, parsed: ParsedProblem) -> sp.Matrix:
        if parsed.matrices:
            return parsed.matrices[0]
        if parsed.vectors:
            return parsed.vectors[0]
        raise ValueError("Nenhuma matriz fornecida.")

    def _h_solve_system(self, p: ParsedProblem) -> AgentResult:
        A = p.extras.get("A") or (p.matrices[0] if p.matrices else None)
        b = p.extras.get("b") or (p.vectors[0] if p.vectors else None)
        if A is None or b is None:
            raise ValueError("solve_system precisa de uma matriz A e um vetor b.")
        return self.solve_linear_system(A, b)

    def _h_determinant(self, p: ParsedProblem) -> AgentResult:
        A = self._first_matrix(p)
        return AgentResult("determinant", value=core.determinant(A))

    def _h_inverse(self, p: ParsedProblem) -> AgentResult:
        A = self._first_matrix(p)
        inv = core.inverse(A)
        valid = validator.check_inverse(A, inv) if self.validate else None
        return AgentResult("inverse", value=inv, valid=valid)

    def _h_transpose(self, p: ParsedProblem) -> AgentResult:
        return AgentResult("transpose", value=core.transpose(self._first_matrix(p)))

    def _h_rank(self, p: ParsedProblem) -> AgentResult:
        return AgentResult("rank", value=analysis.rank(self._first_matrix(p)))

    def _h_nullity(self, p: ParsedProblem) -> AgentResult:
        return AgentResult("nullity", value=analysis.nullity(self._first_matrix(p)))

    def _h_trace(self, p: ParsedProblem) -> AgentResult:
        return AgentResult("trace", value=core.trace(self._first_matrix(p)))

    def _h_eigen(self, p: ParsedProblem) -> AgentResult:
        return self.eigen(self._first_matrix(p))

    def _h_diagonalize(self, p: ParsedProblem) -> AgentResult:
        P, D = solvers.spectral_decomposition(self._first_matrix(p))
        return AgentResult("diagonalize", value=(P, D))

    def _h_lu(self, p: ParsedProblem) -> AgentResult:
        L, U, perm = solvers.lu_decomposition(self._first_matrix(p))
        return AgentResult("lu", value=(L, U, perm))

    def _h_qr(self, p: ParsedProblem) -> AgentResult:
        Q, R = solvers.qr_decomposition(self._first_matrix(p))
        return AgentResult("qr", value=(Q, R))

    def _h_svd(self, p: ParsedProblem) -> AgentResult:
        U, s, Vt = solvers.svd(self._first_matrix(p))
        return AgentResult("svd", value=(U, s, Vt))

    def _h_cholesky(self, p: ParsedProblem) -> AgentResult:
        return AgentResult("cholesky", value=solvers.cholesky(self._first_matrix(p)))

    def _h_gram_schmidt(self, p: ParsedProblem) -> AgentResult:
        vectors = p.vectors or [m for m in p.matrices]
        if not vectors:
            raise ValueError("Forneça vetores para Gram–Schmidt.")
        ortho = analysis.gram_schmidt(vectors, normalize=False)
        valid = validator.check_orthogonality(ortho) if self.validate else None
        return AgentResult("gram_schmidt", value=ortho, valid=valid)

    def _h_project(self, p: ParsedProblem) -> AgentResult:
        if len(p.vectors) < 2:
            raise ValueError("Projeção precisa de pelo menos dois vetores.")
        v = p.vectors[0]
        targets = p.vectors[1:]
        if len(targets) == 1:
            return AgentResult("project", value=analysis.project_vector(v, targets[0]))
        return AgentResult("project", value=analysis.project_onto_subspace(v, targets))

    def _h_rref(self, p: ParsedProblem) -> AgentResult:
        rref, pivots = self._first_matrix(p).rref()
        return AgentResult("rref", value=(rref, pivots))

    def _h_char_poly(self, p: ParsedProblem) -> AgentResult:
        return AgentResult(
            "char_poly", value=analysis.characteristic_polynomial(self._first_matrix(p))
        )

    def _h_is_independent(self, p: ParsedProblem) -> AgentResult:
        vectors = p.vectors or list(p.matrices[0].columnspace()) if p.matrices else []
        if not vectors:
            raise ValueError("Sem vetores para avaliar independência linear.")
        return AgentResult(
            "is_independent", value=analysis.is_linearly_independent(vectors)
        )

    def _h_basis(self, p: ParsedProblem) -> AgentResult:
        if p.vectors:
            return AgentResult("basis", value=analysis.basis_of(p.vectors))
        if p.matrices:
            return AgentResult("basis", value=analysis.column_space(p.matrices[0]))
        raise ValueError("Sem dados para extrair base.")

    def _h_least_squares(self, p: ParsedProblem) -> AgentResult:
        self._require(p, mats=1, vecs=1)
        return AgentResult(
            "least_squares", value=solvers.least_squares(p.matrices[0], p.vectors[0])
        )

    def _h_matmul(self, p: ParsedProblem) -> AgentResult:
        if len(p.matrices) < 2:
            raise ValueError("Produto exige duas matrizes.")
        return AgentResult("matmul", value=core.matmul(p.matrices[0], p.matrices[1]))

    def _h_add(self, p: ParsedProblem) -> AgentResult:
        if len(p.matrices) < 2:
            raise ValueError("Soma exige duas matrizes.")
        return AgentResult("add", value=core.add(p.matrices[0], p.matrices[1]))
