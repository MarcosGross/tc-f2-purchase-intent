# Propensão de Compra em E-commerce

Sistema preditivo que estima a probabilidade de um usuário de e-commerce
concluir uma compra a partir do seu comportamento de navegação na sessão.

**Tech Challenge — Fase 2** · Pós Tech Machine Learning Engineering (FIAP)
**Grupo:** Os Outliers

> As instruções de instalação e execução serão adicionadas na Etapa 4, junto com
> o Dockerfile e o pipeline DVC.

---

## Problema

Lojas online conseguem observar o comportamento de navegação de uma sessão
(páginas visitadas, tempo em cada categoria, taxa de rejeição, origem do
tráfego, proximidade de datas comerciais), mas só descobrem se houve compra ao
final. Antecipar essa intenção permite acionar retenção, recomendação ou
desconto ainda durante a sessão.

Formalizamos o problema como **classificação binária**: dada uma sessão,
prever se ela resultará em compra.

## Dados

[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
— UCI Machine Learning Repository (id 468). Cada linha representa uma sessão de
navegação; a coluna `Revenue` indica se a sessão terminou em compra.

O schema completo (colunas, tipos e balanceamento de classes) será documentado
após o download, no stage de ingestão.

## Foco do projeto

O objetivo desta fase é **engenharia de ML**, não sofisticação de modelagem. Por
isso o modelo é um classificador clássico do Scikit-Learn, e o esforço está em:

| Pilar | Ferramenta |
|---|---|
| Rastreamento de experimentos e registro de modelos | MLflow (+ Model Registry) |
| Versionamento de dados e pipeline reprodutível | DVC |
| Gestão de dependências e lock file | Poetry |
| Empacotamento e execução isolada | Docker |
| Qualidade de código | Ruff, type hints, docstrings |

## Estrutura

```
├── src/purchase_intent/   # código-fonte: um módulo por responsabilidade
├── tests/                 # testes automatizados
├── data/raw/              # dataset bruto        (versionado por DVC)
├── data/processed/        # conjuntos de treino e teste (versionado por DVC)
├── models/                # artefatos de modelo  (versionado por DVC)
├── reports/               # métricas e gráficos  (saída do stage evaluate)
├── configs/params.yaml    # parâmetros de modelagem
└── docs/                  # ML Canvas e model card
```

### Módulos e stages do pipeline

Cada módulo tem uma única responsabilidade, e quase todos correspondem a um
stage do `dvc.yaml`:

| Módulo | Responsabilidade | Stage DVC |
|---|---|---|
| `config.py` | Carrega `.env` e `params.yaml`; fixa a semente | — |
| `data_ingestion.py` | Baixa e lê o dataset bruto | `ingest` |
| `data_split.py` | Divisão estratificada treino/teste | `prepare` |
| `preprocessing.py` | `ColumnTransformer` compartilhado | — |
| `train.py` | Treina o pipeline e registra a run no MLflow | `train` |
| `evaluation.py` | Calcula métricas e grava `reports/metrics.json` | `evaluate` |
| `registry.py` | Promove o modelo no MLflow Model Registry | `register` |
| `inference.py` | Carrega o modelo promovido e prevê | — |

`preprocessing.py` não é um stage por escolha de projeto: ele é importado tanto
por `train.py` quanto por `inference.py` e é serializado dentro do `Pipeline`,
garantindo que treino e inferência apliquem exatamente as mesmas transformações.

## Configuração

Duas fontes de configuração, com responsabilidades distintas:

- **`.env`** — infraestrutura: URI do MLflow, caminhos, nível de log, semente.
  Não é versionado; use o `.env.example` como referência.
- **`configs/params.yaml`** — decisões de modelagem: proporção do split,
  hiperparâmetros, métrica de promoção. É versionado e será declarado como
  `params:` no `dvc.yaml`, de modo que alterá-lo reexecute os stages afetados.

Apenas `config.py` lê essas fontes; os demais módulos recebem a configuração
pronta.

## Roadmap

- [x] **Etapa 1** — Estrutura do repositório e esqueleto dos módulos
- [ ] **Etapa 2** — Poetry, dependências e implementação da ingestão
- [ ] **Etapa 3** — Pipeline DVC (`dvc.yaml`)
- [ ] **Etapa 4** — Docker e instruções de execução
- [ ] **Etapa 5** — Treino, avaliação e tracking no MLflow
- [ ] **Etapa 6** — Model Registry e inferência
