"""Pacote linear_algebra_agent.

Agente especializado em resolver problemas de Álgebra Linear de forma
modular, simbólica/numérica, com explicações passo a passo opcionais.
"""

from .agent import LinearAlgebraAgent
from . import core, solvers, analysis, interface, validator

__all__ = [
    "LinearAlgebraAgent",
    "core",
    "solvers",
    "analysis",
    "interface",
    "validator",
]

__version__ = "0.1.0"
