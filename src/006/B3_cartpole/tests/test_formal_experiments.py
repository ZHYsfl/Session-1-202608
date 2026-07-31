from __future__ import annotations

from argparse import Namespace

import pytest
from formal_runner import (
    DEFAULT_MATRIX,
    completed_training,
    evaluation_is_current,
    evaluation_split,
    formal_hash,
    read_matrix,
    variant_configs,
)
from q_learning import epsilon_by_step


def test_formal_matrix_contains_agreed_experiments() -> None:
    matrix = read_matrix(DEFAULT_MATRIX)
    assert list(matrix["dqn_variants"]) == [
        "dqn_standard",
        "dqn_normalized",
        "double_dqn",
        "double_dqn_fast_exploration",
    ]
    assert list(matrix["q_learning_variants"]) == [
        "q_coarse",
        "q_medium",
        "q_fine",
    ]
    assert matrix["protocol"]["environment_steps"] == 70000
    assert matrix["protocol"]["training_seeds"] == [42, 123, 2026]


def test_variant_configs_change_only_declared_factors() -> None:
    matrix = read_matrix(DEFAULT_MATRIX)
    variants = variant_configs(matrix, formal_hash(DEFAULT_MATRIX))
    standard = variants["dqn_standard"][2]["dqn"]
    full = variants["double_dqn_fast_exploration"][2]["dqn"]
    assert not standard["normalize_state"]
    assert not standard["double_dqn"]
    assert full["normalize_state"]
    assert full["double_dqn"]
    assert variants["q_coarse"][2]["q_learning"]["bins"] == [4, 4, 8, 8]


def test_final_split_requires_explicit_confirmation() -> None:
    matrix = read_matrix(DEFAULT_MATRIX)
    args = Namespace(split="final", confirm_final_test=False)
    with pytest.raises(PermissionError, match="封存"):
        evaluation_split(args, matrix)


def test_q_learning_step_epsilon_decays() -> None:
    config = variant_configs(
        read_matrix(DEFAULT_MATRIX),
        formal_hash(DEFAULT_MATRIX),
    )["q_medium"][2]
    assert epsilon_by_step(config, 0) == pytest.approx(
        config["q_learning"]["epsilon_start"]
    )
    assert epsilon_by_step(config, 70000) < 0.04


def test_completed_training_requires_hash_and_exact_budget(tmp_path) -> None:
    metrics = tmp_path / "training_metrics.json"
    metrics.write_text(
        '{"metadata": {"completed": true, "formal_config_sha256": "abc", '
        '"environment_steps": 70000}}',
        encoding="utf-8",
    )
    assert completed_training(tmp_path, "abc", 70000)
    assert not completed_training(tmp_path, "wrong", 70000)
    assert not completed_training(tmp_path, "abc", 600)


def test_only_matching_evaluation_is_reused(tmp_path) -> None:
    result = tmp_path / "evaluation_development.json"
    result.write_text(
        '{"metadata": {"formal_config_sha256": "abc", "split": "development"}}',
        encoding="utf-8",
    )
    assert evaluation_is_current(result, "abc", "development")
    assert not evaluation_is_current(result, "new", "development")
    assert not evaluation_is_current(result, "abc", "final")
