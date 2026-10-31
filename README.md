# Propensão de Compra em E-commerce

Sistema de classificação binária que estima a probabilidade de uma sessão de
navegação resultar em compra.

**Tech Challenge - Fase 2** · Pós Tech Machine Learning Engineering (FIAP)

**Grupo:** Os Outliers

O projeto reúne ingestão e validação de dados, treinamento com Scikit-Learn,
versionamento com DVC, rastreamento e Model Registry com MLflow e execução em
Docker. O caminho principal deste guia usa Docker Compose para manter os
experimentos e modelos disponíveis após o encerramento dos containers.

## Problema e dados

Uma empresa de e-commerce deseja identificar a propensão de compra a partir do
comportamento de navegação. Cada registro representa uma sessão; o alvo booleano
`Revenue` indica se houve compra. O modelo permite estudar como priorizar ações
de retenção ou ofertas. A integração com uma loja e a previsão durante sessões
em andamento ficam fora do escopo desta implementação.

O dataset é o [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset),
publicado pelo UCI Machine Learning Repository, id 468.

| Característica | Valor |
|---|---|
| Registros | 12.330 sessões |
| Colunas | 17 features e o alvo `Revenue` |
| Sem compra (`False`) | 10.422 sessões, 84,53% |
| Com compra (`True`) | 1.908 sessões, 15,47% |
| Divisão de referência | 9.864 sessões de treino e 2.466 de teste |
| Estratégia de divisão | Estratificada, 80%/20%, seed 42 |

| Tipo no CSV bruto | Colunas |
|---|---|
| `int64` | `Administrative`, `Informational`, `ProductRelated`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| `float64` | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| `object` | `Month`, `VisitorType` |
| `bool` | `Weekend`, `Revenue` |

## Executar do zero com Docker Compose

Os exemplos de terminal usam **PowerShell** e devem ser executados na raiz do
repositório, salvo o comando inicial de clone.

### 1. Preparar o ambiente

Pré-requisitos: Git, Python **3.12**, Poetry **2.5.1** e Docker com containers
Linux e Compose com suporte a `--wait`. O projeto aceita Python 3.11 e 3.12;
3.12 é a versão usada na imagem e na validação de referência.

No Windows, mantenha o Docker Desktop iniciado e configurado para containers
Linux. Após instalar as ferramentas, abra um novo terminal para atualizar o PATH.

```powershell
git clone https://github.com/MarcosGross/tc-f2-purchase-intent.git
Set-Location tc-f2-purchase-intent

poetry env use 3.12
poetry install
poetry check --lock
docker version
docker compose version

if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Para um checkout existente, comece por `poetry env use 3.12` na raiz do projeto.
Se Poetry não localizar Python, informe o caminho do interpretador no comando
`poetry env use`. As dependências de produção e desenvolvimento estão separadas
no `pyproject.toml`, e o `poetry.lock` fixa as versões utilizadas.

### 2. Disponibilizar o dataset antes do build

O repositório distribui o ponteiro DVC do bruto; o CSV deve ser recuperado ou
baixado antes de construir a imagem. Enquanto não houver storage compartilhado
configurado, use o download da fonte oficial:

```powershell
poetry run python -m purchase_intent.data_ingestion
if ($LASTEXITCODE -ne 0) { throw 'Falha no download do dataset.' }

$expectedSha256 = 'b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9'
$rawHash = (Get-FileHash -LiteralPath 'data/raw/online_shoppers_intention.csv' -Algorithm SHA256).Hash
if ($rawHash.ToLowerInvariant() -ne $expectedSha256) {
    throw 'Dataset diferente do snapshot versionado. Confira a fonte antes de continuar.'
}

poetry run dvc commit data/raw/online_shoppers_intention.csv.dvc
if ($LASTEXITCODE -ne 0) { throw 'Falha ao armazenar o snapshot no cache DVC.' }
poetry run dvc status data/raw/online_shoppers_intention.csv.dvc
```

O hash confirma o snapshot esperado antes de gravá-lo no cache. O `dvc commit`
disponibiliza os bytes para um futuro `dvc push`, usando o ponteiro existente.
Quem já tem acesso a um remote pode usar `dvc pull` conforme a seção de
[versionamento e compartilhamento](#versionamento-e-compartilhamento-com-dvc).

### 3. Construir a imagem e executar o pipeline

```powershell
docker build -t tc-f2-purchase-intent:latest .
if ($LASTEXITCODE -ne 0) { throw 'Falha no build da imagem.' }

docker compose up -d --wait mlflow
if ($LASTEXITCODE -ne 0) { throw 'Falha ao iniciar o servidor MLflow.' }

docker compose run --rm pipeline
if ($LASTEXITCODE -ne 0) { throw 'Falha na execução do pipeline.' }

Get-Content -LiteralPath reports/metrics.json
Get-Content -LiteralPath reports/registry.json
```

Abra o [MLflow local](http://localhost:5000) para consultar o experimento
`purchase-intent`, os parâmetros, as métricas e o artefato `model` da run. Em
**Models**, consulte `purchase-intent-classifier` e o alias `champion`.

O serviço `pipeline` monta o checkout em `/app`. Assim, os datasets preparados,
o modelo, os relatórios e as alterações no `dvc.lock` permanecem no host. O banco
SQLite e os artefatos do servidor MLflow ficam no volume nomeado `mlflow-data`,
com o prefixo do projeto Compose, montado em `/mlflow`.

Uma execução sem alterações reutiliza o cache dos estágios. O estágio `register`
é sempre verificado e reutiliza a versão já associada à mesma run. Para executar
todos os estágios novamente e gerar uma nova run:

```powershell
docker compose run --rm pipeline --force --no-run-cache
if ($LASTEXITCODE -ne 0) { throw 'Falha na reprodução completa.' }
```

### 4. Fazer inferência com o modelo promovido

Mantenha o serviço MLflow iniciado. Este exemplo roda dentro do Docker, lê três
sessões do teste e remove o alvo antes de prever. O nome do modelo, o alias e o
limiar vêm da configuração do projeto.

```powershell
@'
from pathlib import Path

import pandas as pd

from purchase_intent.config import load_params, load_settings
from purchase_intent.data_schema import read_dataset
from purchase_intent.inference import load_model, predict, predict_proba

settings = load_settings()
params = load_params(settings.params_path)
alias = params["registry"]["alias"]
model_uri = f"models:/{settings.mlflow_registered_model_name}@{alias}"
threshold = float(params["evaluation"]["decision_threshold"])
sessions = read_dataset(Path("data/processed/test.csv"))
sessions = sessions.drop(columns=[params["dataset"]["target_column"]]).head(3)
model = load_model(model_uri)
result = pd.DataFrame({
    "purchase_probability": predict_proba(model, sessions),
    "predicted_purchase": predict(model, sessions, threshold),
})
print(result.to_string(index=False))
'@ | docker compose run --rm -T --entrypoint python pipeline -
if ($LASTEXITCODE -ne 0) { throw 'Falha na inferência.' }
```

Saída de referência:

```text
 purchase_probability  predicted_purchase
             0.040000               False
             0.146667               False
             0.736667                True
```

Para novas sessões, forneça as mesmas 17 features do treinamento, com nomes e
tipos compatíveis e sem a coluna `Revenue`. O pré-processamento já faz parte do
modelo carregado; não é necessário ajustar outro encoder.

### 5. Encerrar e retomar

```powershell
docker compose down
```

Esse comando remove os containers e a rede, preservando o volume do MLflow e os
arquivos gravados no checkout. Para retomar, execute
`docker compose up -d --wait mlflow` antes de consultar runs ou fazer inferência.
A opção `down -v` remove o volume e, com ele, o banco e os artefatos MLflow desse
ambiente.

## Arquitetura e responsabilidades

```text
src/purchase_intent/   Código organizado por responsabilidade
tests/                Testes de preparação e integridade de dados
data/raw/             CSV bruto e seu ponteiro DVC
data/processed/       Treino, teste e manifesto da divisão
models/               Pipeline Scikit-Learn serializado
reports/              Validação, run ID, métricas e registro do modelo
configs/params.yaml   Seed, split, hiperparâmetros, avaliação e alias
dvc.yaml              Estágios, dependências, parâmetros e saídas
dvc.lock              Identidades da execução versionada
pyproject.toml        Pacote, dependências e ferramentas de qualidade
poetry.lock           Versões resolvidas das dependências
Dockerfile            Imagens de runtime e de testes
docker-compose.yml    Servidor MLflow persistente e execução do pipeline
docs/                 Diretório reservado para documentação complementar
```

O comando padrão da imagem solicita explicitamente esta ordem:

```text
prepare -> validate -> train -> evaluate -> register
```

| Módulo | Responsabilidade | Estágio DVC |
|---|---|---|
| `config.py` | Leitura centralizada de ambiente e parâmetros | - |
| `schema.py` e `data_schema.py` | Schema, hashes e serialização dos dados | - |
| `data_ingestion.py` | Download e leitura do bruto | Inicialização manual |
| `data_split.py` | Divisão estratificada e manifesto das posições | `prepare` |
| `data_validation.py` | Verificação dos dados preparados contra o bruto | `validate` |
| `preprocessing.py` | Construção do `ColumnTransformer` | Incorporado ao modelo |
| `train.py` | Treinamento, parâmetros, modelo e run ID no MLflow | `train` |
| `evaluation.py` | Métricas de teste e atualização da run | `evaluate` |
| `registry.py` | Registro de versão e promoção pelo alias | `register` |
| `inference.py` | Carregamento do modelo e previsões | - |

No grafo atual, `validate` é um ramo de `prepare`; `train` ainda não depende de
seu relatório. A lista explícita de estágios nos comandos deste guia garante a
validação antes do treino. Uma chamada isolada a `dvc repro train` não inclui
essa verificação. Se o pipeline ganhar estágios, atualize também o `ENTRYPOINT`.

A preparação preserva as 18 colunas do bruto. O manifesto registra hashes,
parâmetros e posições originais, distinguindo linhas de valores iguais de uma
posição repetida. A validação confere schema, nulos, tipos, números finitos,
cobertura completa e disjunta, receita do split e preservação de registros e alvo.
Uma falha remove o relatório anterior de sucesso. Os CSVs e JSONs dos dados usam
UTF-8/LF e formato determinístico; a comparação numérica usa tolerância de `1e-12`.

O modelo é um `RandomForestClassifier` com 300 árvores, `class_weight=balanced`
e `random_state=42`. Números são preservados e categorias recebem one-hot
encoding com suporte a categorias desconhecidas. Códigos como `Browser`,
`Region` e `TrafficType` são tratados como categorias nominais. O encoder é
ajustado somente no treino e serializado junto com o classificador.

## Configuração

| Fonte | Responsabilidade |
|---|---|
| `.env` | Infraestrutura, fonte do dataset, caminhos e nível de log; arquivo local ignorado pelo Git |
| `configs/params.yaml` | Seed, alvo, split, hiperparâmetros, limiar, métrica principal e alias; versionado no Git e rastreado pelo DVC |
| `dvc.yaml` | Caminhos explícitos de entrada/saída e dependências de cada estágio |
| `docker-compose.yml` | Serviços, rede, porta, volume e variáveis injetadas nos containers |

O `.env.example` usa `MLFLOW_TRACKING_URI=http://localhost:5000` para clientes
executados no host. O Compose injeta `http://mlflow:5000` no serviço `pipeline`:
dentro de um container, `localhost` aponta para o próprio container. Para
desenvolver com o checkout montado e tracking persistente, use o Compose.

Variáveis já definidas no ambiente prevalecem sobre o `.env`. No Compose, os
nomes de experimento e modelo também são definidos explicitamente no serviço.
A seed efetiva do split e do Random Forest é a chave `seed` do YAML;
`RANDOM_SEED` no `.env` não substitui esse parâmetro. Alterações nos caminhos
dos estágios precisam ser refletidas no `dvc.yaml`.

Após alterar dependências ou o lock, execute `poetry install` e reconstrua a
imagem antes de usar o Compose. Código e parâmetros são lidos da montagem no
Compose; a imagem autocontida precisa ser reconstruída para incorporar mudanças.

## Versionamento e compartilhamento com DVC

| Conteúdo | Controle |
|---|---|
| Código, configurações, `*.dvc`, `dvc.yaml` e `dvc.lock` | Git |
| Bruto, CSVs preparados, manifesto, modelo e relatórios dos estágios | DVC |
| Runs, métricas e versões do Model Registry no servidor | Banco e storage de artefatos MLflow |

O DVC já está inicializado. O bruto pertence ao ponteiro
`data/raw/online_shoppers_intention.csv.dvc`; as demais saídas pertencem aos
estágios do YAML. Use esses contratos ao evoluir o projeto, sem criar outro
produtor para os mesmos caminhos.

### Recuperar uma versão

**Situação atual:** o repositório não distribui um remote DVC. A equipe ainda
precisa escolher e disponibilizar um storage compartilhado. Até lá, o download
com conferência de hash da execução principal permite reconstruir as saídas.

Com um storage realmente acessível à equipe, configure-o no host:

```powershell
poetry run dvc remote add --local -d team-storage 'CAMINHO_OU_URL_DO_STORAGE'
poetry run dvc config --local cache.type copy
poetry run dvc pull
```

Substitua o endereço pelo storage acordado. Se o remote já existe, consulte
`poetry run dvc remote list` e reutilize-o. Instale via Poetry a extensão do
backend escolhido quando necessária. O endereço compartilhável pode ser
versionado em `.dvc/config`; credenciais ficam na configuração local ou no
ambiente. Remotes com caminhos Windows devem ser acessados pelo DVC no host.

O `dvc pull` recupera os arquivos versionados, incluindo o run ID. Ele não
restaura o banco nem o storage MLflow. Se o ambiente MLflow for novo, siga a
[recuperação de runs ausentes](#recuperar-runs-ausentes-no-mlflow).

### Publicar uma versão de dados e código

Após uma mudança intencional no bruto, confira a fonte e atualize o ponteiro
com `poetry run dvc add data/raw/online_shoppers_intention.csv`. Para o snapshot
original baixado pelo guia, o `dvc commit` já o colocou no cache. Execute e
valide o pipeline antes de compartilhar:

```powershell
docker compose run --rm pipeline
if ($LASTEXITCODE -ne 0) { throw 'Falha no pipeline; revise antes de distribuir.' }
poetry run dvc push
git status --short
git diff --check
```

Registre no Git as alterações revisadas de código, parâmetros, ponteiro, YAML e
lock. `git push` distribui código e metadados; `dvc push` distribui os conteúdos.
Confirme a recuperação por outro integrante com um clone limpo e `dvc pull`.
O compartilhamento de runs exige também acesso ao mesmo serviço MLflow e aos
seus artefatos; o Compose deste guia é um ambiente local.

## Recuperar runs ausentes no MLflow

Se o DVC reutilizar `reports/train_run_id.txt`, mas o banco MLflow não contiver
essa run, o registro falhará com `Run ... not found`. Isso pode acontecer ao
trocar o servidor, remover seu volume ou recuperar apenas os arquivos via DVC.

Com o backend de destino disponível, gere uma nova run e reexecute os estágios
que dependem dela, sem reutilizar o cache de execução:

```powershell
docker compose up -d --wait mlflow
if ($LASTEXITCODE -ne 0) { throw 'Servidor MLflow indisponível.' }

docker compose run --rm --entrypoint python pipeline -m dvc repro --single-item train evaluate register --force --no-run-cache
if ($LASTEXITCODE -ne 0) { throw 'Falha ao reconstruir a run e seu registro.' }
```

Esse comando pressupõe que treino e teste já estejam disponíveis. Se também
precisar reconstruir os dados preparados, execute o pipeline completo:

```powershell
docker compose run --rm pipeline --force --no-run-cache
if ($LASTEXITCODE -ne 0) { throw 'Falha na reconstrução completa.' }
```

O procedimento cria uma run no backend escolhido e atualiza run ID, métricas,
registro e lock; não recupera o histórico perdido. Para retomar o histórico
existente, mantenha o banco e o storage de artefatos MLflow correspondentes.

## Outros modos de execução

### Teste de fumaça da imagem

Depois de disponibilizar o bruto e construir a imagem, execute sem montagem:

```powershell
docker run --rm tc-f2-purchase-intent:latest
if ($LASTEXITCODE -ne 0) { throw 'Falha no teste de fumaça.' }
```

A imagem inclui código, parâmetros, receita DVC e snapshot bruto. Usa SQLite
dentro do container e executa os cinco estágios. As runs são rastreadas durante
a execução, mas o `--rm` remove esse banco e as saídas ao final. Use esse modo
para verificar a imagem e o Compose para preservar resultados.

O Dockerfile instala as dependências de produção com Poetry em um estágio
separado, usa uma base Python fixada por digest e executa o runtime como `app`
(uid 1000), com venv Linux em `/opt/venv`. O repositório Git vazio criado na
imagem permite ao DVC operar sem incorporar o histórico do checkout.

### Execução nativa

Com o ambiente Poetry instalado e o servidor Compose iniciado, a configuração
de exemplo permite executar no host:

```powershell
poetry run dvc repro --single-item prepare validate train evaluate register
if ($LASTEXITCODE -ne 0) { throw 'Falha no pipeline nativo.' }
```

Para usar SQLite diretamente no host, selecione esse backend explicitamente.
Ao mudar de backend, force os estágios para criar uma run correspondente:

```powershell
$env:MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
poetry run dvc repro --single-item prepare validate train evaluate register --force --no-run-cache
if ($LASTEXITCODE -ne 0) { throw 'Falha no pipeline com SQLite local.' }
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

A UI desse banco estará em [localhost:5001](http://localhost:5001). Ela mostra
as runs do SQLite local; a UI do Compose mostra as runs armazenadas no volume.
Ao encerrar a UI com `Ctrl+C`, remova a seleção temporária para voltar ao `.env`:

```powershell
Remove-Item Env:MLFLOW_TRACKING_URI
```

## Testes e qualidade

A imagem `test` inclui dependências de desenvolvimento, separadas do runtime:

```powershell
docker build --target test -t tc-f2-purchase-intent:tests .
docker run --rm tc-f2-purchase-intent:tests
docker run --rm tc-f2-purchase-intent:tests ruff check src tests
docker run --rm tc-f2-purchase-intent:tests ruff format --check src tests
poetry check --lock
git diff --check
```

No ambiente nativo, os equivalentes são `poetry run pytest`,
`poetry run ruff check src tests` e `poetry run ruff format --check src tests`.
As fixtures são pequenas, locais e não dependem de download. A suíte atual verifica divisão,
reprodutibilidade, parâmetros, manifesto, integridade e CLI de dados; testes
específicos de modelagem e promoção permanecem como melhoria.

## Resultados e validação

Resultados do baseline na divisão de referência, com limiar de decisão 0,5:

| Métrica | Valor |
|---|---:|
| ROC AUC | 0,9195 |
| Average precision | 0,7172 |
| Accuracy | 0,8885 |
| Precision | 0,6271 |
| Recall | 0,6911 |
| F1 | 0,6575 |

`train` registra hiperparâmetros e modelo; `evaluate` acrescenta as métricas
de teste à mesma run. `register` compara a métrica principal com a versão atual
e aponta `champion` para o candidato quando seu ROC AUC não é inferior. A
repetição do registro da mesma run reutiliza a versão existente. As métricas
descrevem este baseline; não demonstram impacto comercial ou desempenho futuro.

Validação realizada em **28/10/2026**, no código-base `8d10f8d`:

- Instalação pelo Poetry e verificação do lock concluídas.
- Build das imagens de runtime e testes concluído.
- 27 testes aprovados no Windows e no Docker Linux; Ruff e formatação aprovados.
- Os cinco estágios executaram tanto na imagem autocontida quanto no Compose.
- Parâmetros, métricas e artefato do modelo encontrados na run MLflow.
- Versão 1 registrada e promovida para `champion` em ambiente novo.
- Inferência pelo alias validada dentro do Docker.
- Run, modelo e previsões preservados após remover e recriar os containers,
  reutilizando o volume; reexecução com cache manteve a mesma versão.
- Bruto, treino, teste, manifesto, relatório de validação e métricas tiveram
  SHA-256 idêntico entre execução nativa Windows e Docker Linux. Essa comparação
  não inclui os arquivos binários ou metadados do modelo MLflow.

## Entrega do Tech Challenge

| Etapa oficial | Implementação |
|---|---|
| 1 - Clean Code e estrutura | Módulos por responsabilidade, type hints, docstrings e Ruff |
| 2 - Ambiente e dependências | Poetry, lock e configuração externalizada |
| 3 - Containerização e versionamento | Docker, Compose, dataset e pipeline DVC |
| 4 - Modelagem, Registry e entrega | Random Forest, tracking, avaliação, Registry e instruções; vídeo pendente |

- [x] Repositório com histórico de contribuição da equipe.
- [x] Instalação, testes e pipeline validados.
- [x] Modelo, métricas e promoção no MLflow demonstrados.
- [x] Instruções de execução, DVC, recuperação e inferência documentadas.
- [ ] Publicar o vídeo STAR de cinco minutos e inserir seu link abaixo.
- [ ] Conferir o acesso do avaliador ao repositório e ao vídeo antes do envio.

**Vídeo de apresentação:** pendente de publicação. Substituir este texto pelo
link acessível ao avaliador após a gravação.

Roteiro sugerido para o vídeo:

| Intervalo | STAR | Conteúdo |
|---|---|---|
| 0:00-0:40 | Situation | Problema de negócio, dataset e alvo |
| 0:40-1:10 | Task | Objetivos técnicos e restrições |
| 1:10-4:10 | Action | Arquitetura, Poetry, Docker, DVC, pipeline e MLflow |
| 4:10-5:00 | Result | Métricas, modelo promovido, inferência e lições aprendidas |

O linting opcional do enunciado está configurado e validado. O **deploy em
produção na nuvem é opcional e não foi realizado**; o Compose fornece execução
local. O storage DVC compartilhado ainda precisa ser definido pela equipe.
Como melhorias de engenharia, ficam a dependência explícita entre validação e
treino no grafo DVC e a cobertura automatizada de modelagem e Registry.
