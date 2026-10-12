# Propensão de Compra em E-commerce

Sistema preditivo que estima a probabilidade de um usuário de e-commerce
concluir uma compra a partir do seu comportamento de navegação na sessão.

**Tech Challenge — Fase 2** · Pós Tech Machine Learning Engineering (FIAP)
**Grupo:** Os Outliers

As instruções abaixo documentam as Etapas 2 e 3 oficiais. O pipeline completo
também inclui a **Etapa 4 oficial**: treinamento clássico, avaliação, MLflow
Tracking e Model Registry.

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

O CSV bruto contém **12.330 sessões e 18 colunas**. A variável alvo `Revenue` é
booleana: 10.422 sessões sem compra (`False`, 84,53%) e 1.908 com compra
(`True`, 15,47%). A ingestão reportou estes tipos:

| Tipo | Colunas |
|---|---|
| `int64` | `Administrative`, `Informational`, `ProductRelated`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| `float64` | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| `object` | `Month`, `VisitorType` |
| `bool` | `Weekend`, `Revenue` |

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
├── reports/               # validação de dados, métricas e registro do modelo
├── configs/params.yaml    # parâmetros de dados e modelagem
├── dvc.yaml               # pipeline de dados, treino, avaliação e registro
├── dvc.lock               # identidades geradas pela execução do DVC
├── Dockerfile             # ambiente Linux e alvo separado de testes
└── docs/                  # ML Canvas e model card
```

### Módulos e stages do pipeline

Cada módulo tem uma única responsabilidade. O pipeline DVC executa os stages na
ordem `prepare → validate → train → evaluate → register`:

| Módulo | Responsabilidade | Stage DVC |
|---|---|---|
| `config.py` | Carrega `.env` e `params.yaml`; fixa a semente | — |
| `data_ingestion.py` | Baixa e lê o dataset bruto | Inicialização manual; bruto com ponteiro `.dvc` |
| `data_split.py` | Divisão estratificada treino/teste | `prepare` |
| `data_schema.py` | Schema, hashes e serialização compartilhados | — |
| `data_validation.py` | Confere integridade, posições e alvo | `validate` |
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

- **`.env`** — infraestrutura: URI do MLflow, caminhos e nível de log.
  Não é versionado; use o `.env.example` como referência.
- **`configs/params.yaml`** — decisões de modelagem: proporção do split,
  hiperparâmetros, métrica de promoção. É versionado e será declarado como
  `params:` no `dvc.yaml`, de modo que alterá-lo reexecute os stages afetados.
  `seed` e `split.*` configuram a preparação; `model.*`, `evaluation.*` e
  `registry.*` controlam treino, avaliação e promoção. `RANDOM_SEED` no `.env`
  não controla o split. Os caminhos de entrada/saída são explícitos no DVC.

Apenas `config.py` lê essas fontes; os demais módulos recebem a configuração
pronta.

### Ambiente e ingestão (Etapa 2)

Requer Python 3.11 ou 3.12 e Poetry 2.0 ou superior. Na raiz do repositório:

```bash
poetry install
cp .env.example .env  # no PowerShell: Copy-Item .env.example .env
poetry run python -m purchase_intent.data_ingestion
```

O comando baixa o ZIP oficial do UCI, grava o CSV configurado em
`RAW_DATA_PATH` e registra dimensões, tipos de coluna e distribuição de
`Revenue` no log. O arquivo `.env` local não deve ser commitado. O
`poetry.lock` fixa as versões resolvidas para que o grupo instale o mesmo
conjunto de dependências.

### Containerização e versionamento (Etapa 3 oficial)

O enunciado usa `preprocess → train` como **exemplo** de estágios. Para entregar
somente a etapa 3, o fluxo implementado é:

```text
CSV bruto (.dvc) → prepare → train.csv + test.csv + split_manifest.json
                         → validate → reports/data_validation.json
```

`validate` depende do bruto e dos três arquivos preparados. A divisão preserva
todas as 18 colunas, incluindo `Revenue`, sem encoding, scaling, seleção de
features ou treino de modelo. Os nomes train/test identificam apenas conjuntos.

| Arquivo | Controle | Finalidade |
|---|---|---|
| `data/raw/online_shoppers_intention.csv` | DVC, ponteiro individual | Dados originais |
| `data/processed/train.csv` e `test.csv` | Outputs de `prepare` | Registros divididos, sem índice extra |
| `data/processed/split_manifest.json` | Output de `prepare` | Hashes, parâmetros e posições do bruto |
| `reports/data_validation.json` | Output de `validate` | Checks de integridade e distribuição de dados |
| `dvc.yaml`, `dvc.lock`, `*.dvc` e código | Git | Receita e identidades da versão |

O manifesto usa posições originais, permitindo preservar linhas de valores
iguais sem confundi-las com a repetição da mesma posição. O validador confere
hashes, schema, cobertura completa/disjunta e cada registro/alvo; também
recalcula a divisão para conferir a receita. Números no round-trip CSV usam
tolerância `rtol=atol=1e-12`. Saídas são UTF-8/LF, sem horário ou caminho pessoal.
Uma falha retorna código diferente de zero e remove relatório antigo de sucesso.

#### O que cada integrante precisa fazer

| Situação | Ação |
|---|---|
| Vai apenas revisar código | Leia os diffs/README; não precisa baixar dados ou construir imagem |
| Vai executar a etapa 3 | Obtenha esta versão do código, instale o ambiente, disponibilize o bruto e execute Docker conforme abaixo |
| Já tem dados e ambiente dessa versão | Use `dvc pull` quando houver remote acessível e execute o pipeline; não reinicialize DVC |
| Vai continuar o desenvolvimento | Use os arquivos/contratos de dados existentes e acrescente sua responsabilidade; não refaça o versionamento do bruto |

O repositório já tem `.dvc/`; **não rode `dvc init` novamente**. O CSV bruto já
tem um dono (`online_shoppers_intention.csv.dvc`); não crie outro estágio que
produza esse mesmo caminho. Os CSVs preparados e o relatório pertencem ao YAML;
não faça `dvc add` separado nessas saídas.

#### Ambiente

Use Python **3.12**, Poetry **2.5.1** e Docker com containers Linux. No Windows,
Docker Desktop usa WSL 2; abra uma nova janela de PowerShell após instalar as
ferramentas para atualizar o PATH. Execute os comandos na raiz do repositório:

```powershell
poetry env use 3.12
poetry install
poetry check --lock
poetry run dvc version
docker version
```

Se Poetry não localizar Python 3.12, forneça a `poetry env use` o caminho do
interpretador instalado. DVC e NumPy são dependências principais; Ruff e pytest
são dependências de desenvolvimento. `.env`, `.venv`, `poetry.toml`, cache e
configuração pessoal DVC ficam fora do Git.

#### Dados: situação atual e compartilhamento

A validação inicial usou um **remote local**, fora do checkout e configurado com
`--local`. Esse caminho pessoal não é distribuído pelo Git e não fica acessível
a outra máquina. **A equipe ainda precisa escolher um storage compartilhado**
para usar `dvc pull` em computadores diferentes.

Quem já tem acesso a um remote válido deve configurá-lo e recuperar a versão:

```powershell
# Substitua pelo endereço realmente acessível à sua máquina/equipe.
poetry run dvc remote add --local -d team-storage 'CAMINHO_OU_URL_DO_STORAGE'
poetry run dvc config --local cache.type copy
poetry run dvc pull
```

Se o remote já estiver cadastrado, confira `poetry run dvc remote list` e
reutilize-o. Para compartilhar depois, escolham pasta de rede, SSH ou bucket,
instalem a extensão DVC do backend por Poetry se necessária e enviem o conteúdo.
O endereço não secreto pode ficar em `.dvc/config`; credenciais são locais ou
variáveis de ambiente. Outro integrante deve comprovar um clone com `dvc pull`.

**Alternativa enquanto não houver storage compartilhado:** baixe o mesmo bruto
pelo comando da Etapa 2, sem refazer `dvc init` ou atualizar o ponteiro:

```powershell
poetry run python -m purchase_intent.data_ingestion
Get-FileHash -LiteralPath 'data/raw/online_shoppers_intention.csv' -Algorithm SHA256
poetry run dvc status data/raw/online_shoppers_intention.csv.dvc
```

SHA-256 esperado do bruto desta versão:
`b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
Se for diferente, confira a fonte com a equipe antes de atualizar a identidade.
O status do ponteiro deve estar atualizado. Os demais arquivos serão gerados
pelo pipeline. O download é uma alternativa de reprodução, não um substituto
para a decisão de armazenamento compartilhado.

#### Executar no Docker

A imagem tem **dois modos de uso**, ambos a partir do mesmo build:

```powershell
docker build -t tc-f2-purchase-intent:latest .
```

O Dockerfile instala as dependências pelo `poetry.lock` em um estágio separado,
com a base Python fixada por digest. O runtime executa como `app` (uid 1000),
em ambiente Linux com venv em `/opt/venv` — a `.venv` do Windows não é usada.
O comando padrão é `python -m dvc repro --single-item prepare validate train evaluate register`; não há
porta nem servidor a iniciar. Os cinco stages são executados explicitamente para
que uma reprodução forçada não tente refazer o ponteiro do snapshot bruto, o que
alteraria suas permissões em alguns mounts Windows. **Se o grafo for ampliado,
revisem também a lista de stages do `ENTRYPOINT`.**

##### Modo 1 — autocontido (padrão)

A imagem carrega a receita (`dvc.yaml`, `dvc.lock`, `.dvc/`), os parâmetros, o
código e o snapshot bruto (1,07 MB). Um `git init` vazio é criado no build,
porque o DVC exige um repositório para operar — nenhum histórico entra na imagem.

```powershell
docker run --rm tc-f2-purchase-intent:latest
```

Executa o pipeline do zero, sem montar nada. As saídas ficam dentro do container
e são descartadas com `--rm`. É o modo para demonstrar e avaliar.

##### Modo 2 — bind mount (desenvolvimento)

Monta o checkout em `/app`, cobrindo o conteúdo da imagem com o do host:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/app" tc-f2-purchase-intent:latest
if ($LASTEXITCODE -ne 0) { throw 'Falha no pipeline Docker.' }
Get-Content -LiteralPath 'reports/metrics.json'
```

Aqui as saídas são gravadas no checkout e **permanecem depois que o container
sai** — `--rm` remove apenas o container. É o modo para iterar no pipeline. O
`.dockerignore` filtra o build, mas não a pasta montada.

Para comprovar que todos os stages realmente executaram, sem aproveitar
resultados já existentes:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path),target=/app" tc-f2-purchase-intent:latest --force --no-run-cache
if ($LASTEXITCODE -ne 0) { throw 'Falha na reprodução forçada.' }
poetry run dvc status
```

Faça `dvc pull/push` no host quando seu remote for um caminho Windows: `C:\...`
não existe no Linux do container. Mudou o lock? Sincronize o ambiente e
reconstrua a imagem. Mudanças apenas de dados ou parâmetros são lidas da
montagem.

Em uma máquina com dependências nativas funcionais, a execução local equivalente
é `poetry run dvc repro --single-item prepare validate train evaluate register`. Na máquina de validação
da Etapa 3, o Windows bloqueou uma DLL do Scikit-Learn com uma política de
Controle de Aplicativo; o caminho comprovado para executar pipeline e testes foi
o Docker Linux. Não foi comprovada a igualdade dos outputs entre execução nativa
Windows e Linux nessa máquina.

#### Servidor MLflow (docker compose)

O `docker-compose.yml` sobe um servidor MLflow e deixa o pipeline conectado a
ele. Os dois serviços reaproveitam a mesma imagem, para que cliente e servidor
usem a versão de MLflow travada no `poetry.lock`.

```powershell
docker compose up -d mlflow       # servidor em http://localhost:5000
docker compose run --rm pipeline  # pipeline apontando para o servidor
docker compose down               # remove os containers; o volume persiste
```

> **O Model Registry do MLflow exige backend de banco de dados** — não funciona
> com `file:./mlruns`. Duas opções atendem: um SQLite local
> (`sqlite:///mlflow.db`) ou este servidor. O projeto usa o **servidor**, porque
> o SQLite local grava dentro do container e os experimentos se perdem no
> `docker run --rm`; aqui, banco e artefatos ficam no volume nomeado.

O banco e os artefatos ficam no volume nomeado `mlflow-data`, que sobrevive ao
`docker compose down`. O serviço roda como `app`, e por isso `/mlflow` é criado
na imagem já com esse dono: volume nomeado herda a propriedade do diretório que
existe na imagem. O healthcheck usa o próprio Python, porque a base `slim` não
traz `curl`.

#### Testes e qualidade

O alvo `test` inclui ferramentas de desenvolvimento sem adicioná-las ao runtime:

```powershell
docker build --target test -t tc-f2-purchase-intent:tests .
docker run --rm tc-f2-purchase-intent:tests
docker run --rm tc-f2-purchase-intent:tests ruff check src tests
docker run --rm tc-f2-purchase-intent:tests ruff format --check src tests
poetry check --lock
git diff --check
```

Com dependências nativas funcionais, pode-se usar `poetry run pytest`,
`poetry run ruff check src tests` e `poetry run ruff format --check src tests`.
As fixtures são locais, pequenas e sem download de rede.

Validação da etapa 3 em 03/10/2026: 27 testes passaram no Linux/Docker; Ruff,
formatação e lock passaram. A execução real dos dois estágios produziu
9.864/2.466 linhas, mantendo as 18 colunas. Reexecução sem mudança usa cache;
execução forçada preserva hashes por arquivo. A mudança de `test_size` de 0,2
para 0,25 invalida a receita; o retorno a 0,2 recupera a versão de referência.
Um clone local novo, sem CSV ou configuração pessoal, recuperou os cinco
arquivos pelo DVC a partir do storage montado no container e executou ambos os
estágios, preservando os SHA-256 por arquivo. Metadados de permissões (`isexec`)
no lock podem variar entre mounts Windows e arquivos recuperados no Linux;
essa diferença observada não alterou o conteúdo dos dados.

#### Atualizar e distribuir uma versão

Para mudanças intencionais no bruto, confira a fonte, atualize seu ponteiro com
`dvc add`, execute o pipeline e valide. Envie os conteúdos com `dvc push` e
registre código/parâmetros/ponteiro/YAML/lock coerentes no Git. `git push`
distribui código e metadados; `dvc push` distribui os conteúdos. Quem apenas
consome a versão não precisa refazer essa preparação ou mudar os ponteiros.

Branch de implementação: `feat/etapa-3-dvc-docker`, com commits separados por
dependências, dataset, preparação, validação, pipeline, Docker e documentação.
O compartilhamento da branch e sua integração seguem a revisão da equipe.

### Modelagem e MLflow (Etapa 4 oficial)

O baseline usa `RandomForestClassifier`. O pré-processamento preserva as
features numéricas e aplica one-hot encoding nas categorias; códigos como
`Browser`, `Region` e `TrafficType` são categorias nominais, não medidas
ordenadas. O `Pipeline` ajusta o encoder somente no treino e inclui nele o
classificador, evitando divergência entre treino e inferência.

O stage `train` grava o pipeline em `models/trained_pipeline`, cria uma run no
MLflow com hiperparâmetros e modelo, e salva o identificador da run. `evaluate`
usa o conjunto de teste uma vez, registra ROC AUC, average precision, accuracy,
precision, recall e F1, e grava `reports/metrics.json`. `register` registra a
versão no Model Registry e aponta o alias `champion` quando o ROC AUC não é
inferior ao da versão atual.

O Registry exige backend de metadados baseado em banco; `.env.example` usa
SQLite (`sqlite:///mlflow.db`). O banco e os artefatos ficam locais e são
ignorados pelo Git. Para inspecionar as runs após executar o pipeline:

```powershell
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O SQLite local serve para desenvolvimento e demonstração em uma máquina. Para
compartilhar runs e versões registradas entre integrantes, a equipe precisará
de um serviço MLflow e storage de artefatos compartilhados.

Execução de referência desta implementação: ROC AUC `0.9195`, average
precision `0.7172`, accuracy `0.8885`, precision `0.6271`, recall `0.6911` e
F1 `0.6575`, calculados uma vez no teste de 2.466 sessões. Esses valores
documentam este baseline e esta divisão; não são uma meta universal de qualidade.
Uma chamada de inferência pelo alias `champion` também foi verificada em três
sessões de teste.

## Roadmap

Os números abaixo são **marcos internos do repositório**. Os marcos 3 e 4 juntos
correspondem à Etapa 3 oficial do enunciado; os marcos 5 e 6 correspondem à Etapa 4 oficial.

- [x] **Etapa 1** — Estrutura do repositório e esqueleto dos módulos
- [x] **Etapa 2** — Poetry, dependências e implementação da ingestão
- [x] **Etapa 3** — Pipeline DVC (`prepare → validate`, YAML e lock)
- [x] **Etapa 4** — Docker: imagem autocontida, compose e servidor MLflow
- [x] **Etapa 5** — Treino, avaliação e tracking no MLflow
- [x] **Etapa 6** — Model Registry e inferência
