# Redescoberta de Leis Físicas com Regressão Simbólica

Aplicação de machine learning para descobrir automaticamente equações matemáticas a partir de dados de medição de sistemas físicos, sem conhecer a equação.

**Autor:** Luiz Felipe Castro — luiz.castro@usp.br  
**Status:** em desenvolvimento — 

---

## Motivação

Métodos clássicos de ajuste de curvas exigem que o pesquisador proponha a forma funcional da equação antes de ajustá-la aos dados. A regressão simbólica inverte esse processo: dado um conjunto de medições, o algoritmo busca automaticamente a expressão matemática que melhor descreve o fenômeno, equilibrando precisão e simplicidade.

Este projeto aplica essa abordagem a sistemas físicos clássicos, onde a equação real é conhecida — o que permite verificar objetivamente se o algoritmo consegue redescobrir a lei física a partir dos dados.

---

## Pipeline

```
Geração de dados sintéticos (scipy)
        ↓
Regressão simbólica (PySR)
        ↓
Rastreamento de experimentos (MLflow)
        ↓
Versionamento de dados (DVC)
        ↓
Deploy — app interativo (Streamlit)
        ↓
Monitoramento e automação (Evidently + GitHub Actions)
```

---

## Sistemas físicos que vão ser estudados

| Sistema | Equação real | Status |
|---|---|---|
| Queda livre | y = v₀t - ½gt² | pendente |
| Oscilador harmônico | x(t) = A·cos(ωt + φ) | pendente |
| Pêndulo amortecido | θ'' + (b/m)θ' + (g/L)θ = 0 | pendente |
| Resfriamento de Newton | T(t) = T∞ + (T₀ - T∞)·e^(-kt) | pendente |

---

## Estrutura do repositório

```
symbolic_regression_physics/
├── data/
│   └── raw/               ← datasets sintéticos (rastreados pelo DVC)
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_symbolic_regression.ipynb
│   └── 03_analysis.ipynb
├── app/
│   └── streamlit_app.py   ← interface interativa
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

```bash
git clone https://github.com/seu-usuario/symbolic_regression_physics.git
cd symbolic_regression_physics
pip install -r requirements.txt
```

Execute os notebooks em ordem numérica. O app Streamlit pode ser iniciado com:

```bash
streamlit run app/streamlit_app.py
```

---

## Tecnologias

- **PySR** — regressão simbólica via algoritmo genético
- **scipy** — geração de dados e solução numérica de EDOs
- **MLflow** — rastreamento de experimentos
- **DVC** — versionamento de dados
- **Streamlit** — interface interativa
- **Evidently AI** — monitoramento de qualidade
- **GitHub Actions** — automação do pipeline
- **Docker** — containerização
