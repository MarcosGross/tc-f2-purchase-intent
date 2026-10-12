# syntax=docker/dockerfile:1
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016 AS base

FROM base AS builder
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install --no-cache-dir poetry==2.5.1
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
RUN /opt/poetry/bin/poetry sync --only main --no-root

FROM builder AS test
RUN /opt/poetry/bin/poetry sync --no-root
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && git config --system --add safe.directory /app
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Camadas ordenadas da que muda menos para a que muda mais, para preservar cache.
# Receita e identidades do pipeline.
COPY --chown=app:app pyproject.toml poetry.lock dvc.yaml dvc.lock ./
COPY --chown=app:app .dvc/ ./.dvc/
# Entrada dos stages (snapshot bruto, ~1 MB).
COPY --chown=app:app data/raw/ ./data/raw/
# Configuração e código.
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app src/ ./src/

# As saídas são geradas aqui dentro: os diretórios precisam existir e pertencer ao app.
RUN mkdir -p data/processed reports && chown -R app:app /app
# Ponto de montagem do volume do MLflow. Criado aqui com dono app para que
# o volume nomeado herde a propriedade e o servidor (não-root) consiga escrever.
RUN mkdir -p /mlflow/artifacts && chown -R app:app /mlflow
USER app
# O DVC exige um repositório git. Um init vazio basta: nenhum histórico entra na imagem.
RUN git init -q .
# O snapshot bruto é entrada; a lista explícita reproduz todo o pipeline de ML sem
# tentar refazer o ponteiro do bruto numa reprodução forçada.
ENTRYPOINT ["python", "-m", "dvc", "repro", "--single-item", "prepare", "validate", "train", "evaluate", "register"]
