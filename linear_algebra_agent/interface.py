"""interface.py — Camada de linguagem natural.

Recebe um enunciado em texto e o converte em um *comando estruturado*
:class:`ParsedProblem` que o :class:`~.agent.LinearAlgebraAgent` sabe
executar.

O parser segue uma estratégia leve baseada em expressões regulares e
heurísticas; é deliberadamente extensível: novos verbos podem ser
registrados em :data:`OPERATION_KEYWORDS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp


# ---------------------------------------------------------------------------
# Estrutura do problema interpretado
# ---------------------------------------------------------------------------

@dataclass
class ParsedProblem:
    """Representação estruturada de um problema."""

    operation: str
    matrices: List[sp.Matrix] = field(default_factory=list)
    vectors: List[sp.Matrix] = field(default_factory=list)
    scalars: List[sp.Expr] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ParsedProblem(op={self.operation!r}, "
            f"matrices={len(self.matrices)}, vectors={len(self.vectors)}, "
            f"scalars={self.scalars}, extras={self.extras})"
        )


# ---------------------------------------------------------------------------
# Vocabulário — palavras-chave → operação canônica
# ---------------------------------------------------------------------------

OPERATION_KEYWORDS: Dict[str, List[str]] = {
    "solve_system": [
        "resolver sistema", "resolva o sistema", "solucionar", "solve system",
        "ax = b", "ax=b", "sistema linear",
    ],
    "determinant": ["determinante", "det(", "det "],
    "inverse": ["inversa", "inverse", "matriz inversa"],
    "transpose": ["transposta", "transpose"],
    "rank": ["posto", "rank"],
    "nullity": ["nulidade", "nullity"],
    "trace": ["traço", "traco", "trace"],
    "eigen": [
        "autovalor", "autovalores", "autovetor", "autovetores",
        "eigenvalue", "eigenvector", "espectro",
    ],
    "diagonalize": ["diagonaliz", "decomposição espectral", "decomposicao espectral"],
    "lu": ["decomposição lu", "decomposicao lu", "lu decomposition", "fatorar lu"],
    "qr": ["decomposição qr", "decomposicao qr", "qr decomposition", "fatoração qr", "fatoracao qr"],
    "svd": ["svd", "decomposição svd", "valores singulares"],
    "cholesky": ["cholesky"],
    "gram_schmidt": ["gram-schmidt", "ortonormaliz", "ortogonaliz"],
    "project": ["projeção", "projecao", "projeta", "project"],
    "rref": ["forma escalonada", "rref", "escalonad"],
    "char_poly": ["polinômio característico", "polinomio caracteristico", "characteristic polynomial"],
    "is_independent": [
        "linearmente independente", "independência linear", "independencia linear",
    ],
    "basis": ["base de", "extrair base", "encontrar base"],
    "least_squares": ["mínimos quadrados", "minimos quadrados", "least squares"],
    "matmul": ["produto de matrizes", "multiplicar matrizes", "a * b", "a*b"],
    "add": ["somar matrizes", "soma de matrizes", "a + b"],
}


# ---------------------------------------------------------------------------
# Extração de matrizes / vetores
# ---------------------------------------------------------------------------

# Aceita formatos como:
#   [[1,2],[3,4]]
#   [1, 2, 3]
#   {{1,2},{3,4}}
_MATRIX_PATTERNS = [
    re.compile(r"\[\s*\[[^\[\]]*?\](?:\s*,\s*\[[^\[\]]*?\])*\s*\]"),  # listas aninhadas
    re.compile(r"\{\s*\{[^{}]*?\}(?:\s*,\s*\{[^{}]*?\})*\s*\}"),       # chaves aninhadas
    re.compile(r"\[[^\[\]]+?\]"),                                       # vetor simples
]


def _normalize_braces(text: str) -> str:
    return text.replace("{", "[").replace("}", "]")


def _parse_one(token: str) -> Optional[sp.Matrix]:
    """Tenta converter um trecho em ``sympy.Matrix``."""
    token = _normalize_braces(token).strip()
    try:
        parsed = sp.sympify(
            token,
            locals={},
            convert_xor=True,
        )
    except (sp.SympifyError, SyntaxError, TypeError):
        return None
    try:
        if isinstance(parsed, sp.MatrixBase):
            return sp.Matrix(parsed)
        if isinstance(parsed, (list, tuple, sp.Tuple)):
            data = list(parsed)
            if not data:
                return None
            if isinstance(data[0], (list, tuple, sp.Tuple)):
                rows = [list(row) for row in data]
                widths = {len(r) for r in rows}
                if len(widths) != 1:
                    return None
                return sp.Matrix(rows)
            return sp.Matrix(list(data))
    except (TypeError, ValueError):
        return None
    return None


def extract_arrays(text: str) -> Tuple[List[sp.Matrix], List[sp.Matrix], str]:
    """Extrai matrizes e vetores de ``text``.

    Retorna ``(matrices, vectors, texto_residual)``.
    """
    matrices: List[sp.Matrix] = []
    vectors: List[sp.Matrix] = []
    cleaned = text
    seen_spans: List[Tuple[int, int]] = []

    for pattern in _MATRIX_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                continue
            obj = _parse_one(match.group(0))
            if obj is None:
                continue
            seen_spans.append(span)
            r, c = obj.shape
            if r == 1 or c == 1:
                vectors.append(obj if c == 1 else obj.T)
            else:
                matrices.append(obj)
            cleaned = cleaned.replace(match.group(0), " ", 1)
    return matrices, vectors, cleaned


def extract_scalars(text: str) -> List[sp.Expr]:
    """Extrai escalares numéricos isolados (inteiros e decimais)."""
    tokens = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)
    return [sp.sympify(t) for t in tokens]


# ---------------------------------------------------------------------------
# Detecção da operação
# ---------------------------------------------------------------------------

def detect_operation(text: str) -> str:
    lowered = text.lower()
    best: Tuple[str, int] = ("unknown", -1)
    for op, kws in OPERATION_KEYWORDS.items():
        for kw in kws:
            idx = lowered.find(kw)
            if idx >= 0:
                # prioriza palavras-chave mais específicas (mais longas)
                score = len(kw) * 1000 - idx
                if score > best[1]:
                    best = (op, score)
    return best[0]


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def parse_problem(text: str) -> ParsedProblem:
    """Converte texto em ``ParsedProblem``."""
    if not text or not text.strip():
        raise ValueError("Enunciado vazio.")
    matrices, vectors, residual = extract_arrays(text)
    scalars = extract_scalars(residual)
    op = detect_operation(text)
    extras: Dict[str, Any] = {}

    if op == "solve_system" and matrices and vectors:
        extras["A"] = matrices[0]
        extras["b"] = vectors[0]
    elif op == "project" and len(vectors) >= 2:
        extras["v"] = vectors[0]
        extras["onto"] = vectors[1:] if len(vectors) > 2 else vectors[1]

    return ParsedProblem(
        operation=op,
        matrices=matrices,
        vectors=vectors,
        scalars=scalars,
        extras=extras,
        raw=text,
    )


def register_operation(name: str, keywords: List[str]) -> None:
    """Permite estender o vocabulário em tempo de execução."""
    OPERATION_KEYWORDS[name] = keywords
