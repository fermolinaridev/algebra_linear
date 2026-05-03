# algebra_linear

Agente especializado em resolver, automaticamente, qualquer problema de
**Álgebra Linear**. Implementado em Python, com arquitetura modular e foco
em precisão matemática (cálculo simbólico via `sympy`, com fallback
numérico em `numpy`), clareza de raciocínio e extensibilidade.

```
algebra_linear/
├── linear_algebra_agent/
│   ├── __init__.py
│   ├── agent.py         # Classe principal LinearAlgebraAgent
│   ├── core.py          # Operações fundamentais (matrizes/vetores)
│   ├── solvers.py       # Sistemas lineares e decomposições
│   ├── analysis.py      # Posto, bases, projeções, Gram–Schmidt
│   ├── interface.py     # Parser de linguagem natural
│   └── validator.py     # Verificação e explicações passo a passo
├── tests/               # Testes pytest para cada módulo
├── examples/            # Exemplos práticos prontos para rodar
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

Requer Python ≥ 3.10. Dependências: `numpy`, `sympy`, `pytest`.

## Uso rápido

```python
from linear_algebra_agent import LinearAlgebraAgent

agent = LinearAlgebraAgent(explain=True, validate=True)

# Linguagem natural
res = agent.solve("Resolver sistema [[2,1],[1,3]] com b = [5, 10]")
print(res.pretty())

# API direta
import sympy as sp
res = agent.eigen(sp.Matrix([[2, 0], [0, 3]]))
print(res.value)
```

A entrada principal é `agent.solve(problem: str)`, que aceita texto
contendo:

* matrizes em formato `[[a,b],[c,d]]` ou `{{a,b},{c,d}}`;
* vetores `[x, y, z]`;
* palavras-chave em português ou inglês — ver
  `linear_algebra_agent/interface.py::OPERATION_KEYWORDS`.

## Capacidades

| Categoria | Implementação |
|-----------|---------------|
| Sistemas lineares (diretos) | `solvers.solve_system` (Gauss-Jordan, LU, Cramer) |
| Sistemas lineares (iterativos) | `solvers.jacobi`, `solvers.gauss_seidel` |
| Operações matriciais | `core.add`, `matmul`, `transpose`, `inverse`, `power`, `pseudoinverse` |
| Operações vetoriais | `core.dot`, `cross`, `norm`, `normalize`, `angle` |
| Determinante / traço / posto | `core.determinant`, `trace`, `analysis.rank` |
| Decomposições | `solvers.lu_decomposition`, `qr_decomposition`, `cholesky`, `svd`, `spectral_decomposition` |
| Espaços fundamentais | `analysis.column_space`, `row_space`, `null_space`, `left_null_space` |
| Bases e mudanças de base | `analysis.basis_of`, `coordinates_in_basis`, `change_of_basis_matrix` |
| Independência linear | `analysis.is_linearly_independent` |
| Ortogonalização / projeções | `analysis.gram_schmidt`, `project_vector`, `project_onto_subspace` |
| Autovalores / autovetores | `solvers.eigen`, `analysis.is_diagonalizable` |
| Polinômios característico/minimal | `analysis.characteristic_polynomial`, `minimal_polynomial` |
| Mínimos quadrados | `solvers.least_squares` |
| Manipulação simbólica | Toda a stack opera sobre `sympy.Matrix` |

## Precisão e robustez

* Cálculos simbólicos exatos por padrão (sem perda de precisão).
* Conversão automática para `numpy` em rotinas numéricas (SVD, métodos
  iterativos), com tolerância controlada via parâmetro `tol`.
* Erros matemáticos são reportados via `ValueError` com mensagem clara
  (matriz singular, sistema inconsistente, dimensões incompatíveis).
* O motor `validator` confere consistência de cada cálculo: solução de
  sistema, par autovalor/autovetor, ortogonalidade, inversa.

## Explicações passo a passo

Quando o agente é instanciado com `explain=True`, sistemas lineares e
métodos diretos registram um log de raciocínio em `result.explanation`.
Use `result.pretty()` para imprimir o resultado formatado.

## Executando os exemplos

```bash
python examples/01_linear_systems.py
python examples/02_eigen.py
python examples/03_bases_and_projections.py
python examples/04_decompositions.py
```

## Rodando os testes

```bash
pytest -q
```

## Como estender o agente

Há dois pontos de extensão principais:

1. **Vocabulário** — registre novas palavras-chave que mapeiam para
   operações:

   ```python
   from linear_algebra_agent.interface import register_operation
   register_operation("rotacao_2d", ["rotacionar", "rotation"])
   ```

2. **Handler** — registre a função que executa a nova operação:

   ```python
   from linear_algebra_agent import LinearAlgebraAgent
   from linear_algebra_agent.agent import AgentResult

   def rotacao_2d_handler(parsed):
       theta = parsed.scalars[0]
       import sympy as sp
       R = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                      [sp.sin(theta),  sp.cos(theta)]])
       return AgentResult("rotacao_2d", value=R * parsed.vectors[0])

   agent = LinearAlgebraAgent()
   agent.register_handler("rotacao_2d", rotacao_2d_handler)
   ```

Para introduzir um novo *módulo de domínio* (por exemplo, espaços com
produto interno generalizado), siga o padrão de `analysis.py`: funções
puras que recebem `MatrixLike` (ver `core.MatrixLike`) e retornam
`sympy.Matrix` ou escalares simbólicos. Em seguida, exponha-as no
`__init__.py` e crie handlers em `agent.py` apontando para a operação.

## Licença

MIT.
