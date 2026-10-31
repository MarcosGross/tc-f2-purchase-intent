"""Testes de pré-processamento, treinamento multi-algoritmo e balanceamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purchase_intent.data_schema import EXPECTED_COLUMNS, NUMERIC_COLUMNS
from purchase_intent.data_split import split_features_target, stratified_split
from purchase_intent.evaluation import evaluate_pipeline
from purchase_intent.inference import predict, predict_proba
from purchase_intent.preprocessing import build_preprocessor, infer_feature_types
from purchase_intent.train import (
    SUPPORTED_MODELS,
    build_estimator,
    build_pipeline,
    train_pipeline,
)


@pytest.fixture
def imbalanced_frame() -> pd.DataFrame:
    """Cria um dataset sintético desbalanceado (~85% False, ~15% True) com sinal claro."""
    rng = np.random.default_rng(42)
    rows = 200
    # 170 sessões sem compra (False) e 30 com compra (True)
    target = np.array([False] * 170 + [True] * 30)

    data: dict[str, object] = {}
    for name in NUMERIC_COLUMNS:
        # Sessões com compra têm valores ligeiramente maiores mas com sobreposição
        base = rng.normal(loc=0.0, scale=1.0, size=rows)
        signal = np.where(target, 1.1, 0.0)
        data[name] = np.abs(base + signal)

    data["Month"] = rng.choice(["Feb", "Mar", "May", "Nov"], size=rows).tolist()
    data["OperatingSystems"] = rng.choice([1, 2, 3], size=rows).tolist()
    data["Browser"] = rng.choice([1, 2], size=rows).tolist()
    data["Region"] = rng.choice([1, 2, 3], size=rows).tolist()
    data["TrafficType"] = rng.choice([1, 2, 4], size=rows).tolist()
    data["VisitorType"] = rng.choice(["Returning_Visitor", "New_Visitor"], size=rows).tolist()
    data["Weekend"] = rng.choice([False, True], size=rows).tolist()
    data["Revenue"] = target.tolist()

    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)


@pytest.mark.parametrize("model_name", sorted(SUPPORTED_MODELS))
@pytest.mark.parametrize("balanced", [False, True])
def test_all_models_train_evaluate_and_predict(
    imbalanced_frame: pd.DataFrame,
    model_name: str,
    balanced: bool,
):
    """Garante que os 3 algoritmos treinam e geram métricas válidas com e sem balanceamento."""
    features, target = split_features_target(imbalanced_frame, "Revenue")
    splits = stratified_split(features, target, test_size=0.25, seed=42)
    numeric, categorical = infer_feature_types(splits.x_train)

    preprocessor = build_preprocessor(numeric, categorical)
    estimator = build_estimator(model_name, {}, seed=42, balanced=balanced)
    pipeline = train_pipeline(
        build_pipeline(preprocessor, estimator), splits.x_train, splits.y_train
    )

    metrics = evaluate_pipeline(pipeline, splits.x_test, splits.y_test, threshold=0.5)
    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    probs = predict_proba(pipeline, splits.x_test)
    preds = predict(pipeline, splits.x_test, threshold=0.5)
    assert len(probs) == len(preds) == len(splits.x_test)
    assert preds.dtype == bool


@pytest.mark.parametrize("model_name", sorted(SUPPORTED_MODELS))
def test_balancing_increases_or_preserves_minority_recall(
    imbalanced_frame: pd.DataFrame,
    model_name: str,
):
    """Comprova que ativar o balanceamento não reduz o recall da classe minoritária."""
    features, target = split_features_target(imbalanced_frame, "Revenue")
    splits = stratified_split(features, target, test_size=0.3, seed=42)
    numeric, categorical = infer_feature_types(splits.x_train)

    unbalanced_pipe = train_pipeline(
        build_pipeline(
            build_preprocessor(numeric, categorical),
            build_estimator(model_name, {}, seed=42, balanced=False),
        ),
        splits.x_train,
        splits.y_train,
    )
    balanced_pipe = train_pipeline(
        build_pipeline(
            build_preprocessor(numeric, categorical),
            build_estimator(model_name, {}, seed=42, balanced=True),
        ),
        splits.x_train,
        splits.y_train,
    )

    metrics_unbalanced = evaluate_pipeline(unbalanced_pipe, splits.x_test, splits.y_test, 0.5)
    metrics_balanced = evaluate_pipeline(balanced_pipe, splits.x_test, splits.y_test, 0.5)

    assert metrics_balanced["recall"] >= metrics_unbalanced["recall"]


def test_unknown_model_is_rejected():
    """Garante erro explícito quando um nome de modelo inválido é configurado."""
    with pytest.raises(ValueError, match="Modelo não suportado"):
        build_estimator("svm_inexistente", {}, seed=42)
