"""Tests for TD RL evaluation configuration."""

from fle.env.gym_env.td_config import TDScenarioConfig
from fle.rl.config import EvalConfig
from fle.rl.eval import _parse_args, _scenario_config


def test_eval_scenario_uses_training_difficulty_preset():
    cfg = EvalConfig(model_path="model.zip", difficulty="hard")

    scenario = _scenario_config(cfg)

    assert scenario.turret_deletion_percentage == TDScenarioConfig.HARD.turret_deletion_percentage
    assert scenario.starting_inventory == TDScenarioConfig.HARD.starting_inventory


def test_eval_scenario_overrides_optional_difficulty_knobs():
    cfg = EvalConfig(
        model_path="model.zip",
        difficulty="easy",
        evolution_factor=0.75,
        max_unit_group_size=33,
        game_speed=1.5,
    )

    scenario = _scenario_config(cfg)

    assert scenario.turret_deletion_percentage == TDScenarioConfig.EASY.turret_deletion_percentage
    assert scenario.evolution_factor == 0.75
    assert scenario.max_unit_group_size == 33
    assert scenario.game_speed == 1.5


def test_parse_args_exposes_difficulty_and_stochastic_eval():
    cfg = _parse_args(
        [
            "--model",
            "model.zip",
            "--difficulty",
            "easy",
            "--no-deterministic",
        ]
    )

    assert cfg.difficulty == "easy"
    assert cfg.deterministic is False
