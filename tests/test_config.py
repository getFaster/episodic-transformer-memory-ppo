from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from episodic_moba_ppo.config import PretrainedEvalConfig, TrainConfig, load_config


ROOT = Path(__file__).parents[1]


def load_raw(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("pretrained_eval.yaml", PretrainedEvalConfig),
        ("trxl_command40.yaml", TrainConfig),
        ("trxl_moba_command40.yaml", TrainConfig),
    ],
)
def test_shipped_configs_validate(name: str, expected_type: type) -> None:
    config = load_config(ROOT / "configs" / name)
    assert isinstance(config, expected_type)
    assert config.provenance.verify_checkpoint(ROOT).name == "mortar_mayhem_grid_trxl.nn"


def test_unknown_fields_are_rejected_recursively() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["attention"]["typo_budget"] = 256
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainConfig.model_validate(raw)


@pytest.mark.parametrize("model_seed", [0, 4, 10000])
def test_only_paired_model_seeds_are_accepted(model_seed: int) -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["seeds"]["model"] = model_seed
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_training_environment_seed_pool_cannot_enter_reserved_range() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["seeds"]["environment_start"] = 9999
    raw["seeds"]["environment_count"] = 2
    raw["environment"]["seed_start"] = 9999
    raw["environment"]["seed_count"] = 2
    with pytest.raises(ValidationError, match="0..9999"):
        TrainConfig.model_validate(raw)


def test_attention_budget_relationship_is_enforced() -> None:
    raw = load_raw("trxl_moba_command40.yaml")
    raw["attention"]["retrieval"]["retrieved_blocks"] = 7
    with pytest.raises(ValidationError, match=r"block_size \* retrieved_blocks"):
        TrainConfig.model_validate(raw)


def test_arm_specific_retrieval_contract_is_enforced() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["attention"]["retrieval"]["enabled"] = True
    with pytest.raises(ValidationError, match="trxl must disable retrieval"):
        TrainConfig.model_validate(raw)


def test_training_preserves_checkpoint_explosion_delay() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["environment"]["explosion_delay"] = 6
    with pytest.raises(ValidationError, match="checkpoint explosion_delay 5"):
        TrainConfig.model_validate(raw)


def test_environment_step_budget_is_derived_from_updates() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["ppo"]["environment_steps"] = 1_015_808
    with pytest.raises(ValidationError, match=r"updates \* 16,384"):
        TrainConfig.model_validate(raw)


def test_extension_requires_gate_artifact_and_terminal_milestone() -> None:
    raw = load_raw("trxl_command40.yaml")
    raw["ppo"]["updates"] = 62
    raw["ppo"]["environment_steps"] = 1_015_808
    with pytest.raises(ValidationError, match="extension_gate_artifact"):
        TrainConfig.model_validate(raw)

    raw["ppo"]["extension_gate_artifact"] = "results/extension_gate.json"
    with pytest.raises(ValidationError, match="terminal PPO update"):
        TrainConfig.model_validate(raw)

    raw["checkpointing"]["milestone_updates"] = [31, 62]
    config = TrainConfig.model_validate(raw)
    assert config.ppo.updates == 62


def test_baseline_seed_protocol_is_locked() -> None:
    raw = load_raw("pretrained_eval.yaml")
    raw["evaluation"]["seeds"]["action_rng_repeats"] = 3
    with pytest.raises(ValidationError, match="two paired"):
        PretrainedEvalConfig.model_validate(raw)


def test_strict_types_do_not_coerce_seed_strings() -> None:
    raw = deepcopy(load_raw("trxl_command40.yaml"))
    raw["seeds"]["model"] = "1"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)
