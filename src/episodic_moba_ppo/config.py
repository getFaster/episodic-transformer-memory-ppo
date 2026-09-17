"""Strict, versioned configuration models for all experiment entry points."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Arm: TypeAlias = Literal["trxl", "trxl_moba"]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
PRIMARY_COMMAND_COUNTS = [10, 20, 30, 40, 50, 60, 80]


class StrictModel(BaseModel):
    """Base model that rejects coercion, mutation, and unknown YAML fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ProvenanceConfig(StrictModel):
    upstream_repo: str
    upstream_commit: str
    checkpoint_path: str
    checkpoint_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> ProvenanceConfig:
        if not _COMMIT_RE.fullmatch(self.upstream_commit):
            raise ValueError("upstream_commit must be a lowercase 40-character Git hash")
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase 64-character SHA-256")
        return self

    def verify_checkpoint(self, repo_root: str | Path = ".") -> Path:
        """Resolve and hash-check the configured checkpoint before deserialization."""
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = Path(repo_root) / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint}")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != self.checkpoint_sha256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {self.checkpoint_sha256}, got {digest}"
            )
        return checkpoint


class EnvironmentConfig(StrictModel):
    """Historical Mortar Mayhem environment contract.

    This model is deliberately kept separate from :class:`MiniGridEnvironmentConfig`.
    Existing Mortar YAML documents remain both valid and semantically unchanged.
    """

    name: Literal["MortarMayhem-Grid-v0"]
    command_count: Annotated[int, Field(ge=1)]
    seed_start: Annotated[int, Field(ge=0)]
    seed_count: Annotated[int, Field(ge=1)]
    agent_scale: Annotated[float, Field(gt=0)]
    arena_size: Annotated[int, Field(ge=1)]
    allowed_commands: Annotated[int, Field(ge=1)]
    explosion_duration: Annotated[int, Field(ge=0)]
    explosion_delay: Annotated[int, Field(ge=0)]
    reward_command_failure: float
    reward_command_success: float
    reward_episode_success: float


class TransformerConfig(StrictModel):
    num_blocks: Literal[3]
    embed_dim: Literal[384]
    num_heads: Literal[4]
    memory_length: Literal[118]
    positional_encoding: Literal["relative"]
    layer_norm: Literal["pre"]
    gtrxl: Literal[False]
    gtrxl_bias: Literal[0.0]


class MiniGridEnvironmentConfig(StrictModel):
    """Locked MiniGrid Memory S9 delay-sweep task specification."""

    name: Literal["MiniGrid-MemoryS9-v0"]
    scenario: Literal["S9"]
    backend: Literal["minigrid==3.1.0"]
    legacy_fallback_backend: Literal["gym-minigrid==1.0.2"]
    delay_conditions: list[Literal[32, 64, 96, 128]]
    task_rng_seed: Annotated[int, Field(ge=0)]
    seed_start: Annotated[int, Field(ge=0)]
    seed_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_delay_sweep(self) -> MiniGridEnvironmentConfig:
        if self.delay_conditions != [32, 64, 96, 128]:
            raise ValueError("MiniGrid delay_conditions must be [32, 64, 96, 128]")
        return self


class MiniGridTransformerConfig(StrictModel):
    """The exact architecture serialized in models/minigrid.nn."""

    num_blocks: Literal[3]
    embed_dim: Literal[384]
    num_heads: Literal[4]
    memory_length: Literal[64]
    positional_encoding: Literal["relative"]
    layer_norm: Literal["post"]
    gtrxl: Literal[False]
    gtrxl_bias: Literal[0.0]


class LoraConfig(StrictModel):
    backend: Literal["peft"]
    enabled: bool
    targets: list[Literal["q", "k", "v", "o"]]
    rank: Literal[8]
    alpha: Literal[16]
    dropout: Literal[0.0]
    a_init: Literal["random"]
    b_init: Literal["zeros"]

    @model_validator(mode="after")
    def validate_targets(self) -> LoraConfig:
        if self.targets != ["q", "k", "v", "o"]:
            raise ValueError("LoRA targets must be exactly [q, k, v, o] in every layer")
        return self


class RetrievalConfig(StrictModel):
    enabled: bool
    block_size: Annotated[int, Field(ge=1)]
    retrieved_blocks: Annotated[int, Field(ge=0)]
    retrieved_tokens: Annotated[int, Field(ge=0)]
    score_reduction: Literal["max_head"]
    tie_break: Literal["chronological_block_index"]

    @model_validator(mode="after")
    def validate_capacity(self) -> RetrievalConfig:
        expected = self.block_size * self.retrieved_blocks
        if self.retrieved_tokens != expected:
            raise ValueError(
                "retrieved_tokens must equal block_size * retrieved_blocks "
                f"({expected})"
            )
        return self


class AttentionConfig(StrictModel):
    budget: Annotated[int, Field(ge=1)]
    dense_recent: Annotated[int, Field(ge=1)]
    search_horizon: Annotated[int, Field(ge=1)]
    retrieval: RetrievalConfig

    @model_validator(mode="after")
    def validate_budget(self) -> AttentionConfig:
        if self.dense_recent > self.budget:
            raise ValueError("dense_recent cannot exceed attention budget")
        if self.search_horizon < self.budget:
            raise ValueError("search_horizon cannot be smaller than attention budget")
        attended = self.dense_recent + self.retrieval.retrieved_tokens
        if attended != self.budget:
            raise ValueError(
                "dense_recent + retrieved_tokens must equal attention budget "
                f"({attended} != {self.budget})"
            )
        return self


class PPOConfig(StrictModel):
    updates: Literal[31, 62]
    environment_steps: Literal[507904, 1015808]
    workers: Literal[32]
    worker_steps: Literal[512]
    epochs: Annotated[int, Field(ge=1)]
    minibatches: Literal[8]
    effective_minibatch_size: Literal[2048]
    microbatch_size: Literal[256]
    gamma: Annotated[float, Field(gt=0, le=1)]
    gae_lambda: Annotated[float, Field(ge=0, le=1)]
    clip_range: Annotated[float, Field(gt=0)]
    entropy_beta_initial: Annotated[float, Field(ge=0)]
    entropy_beta_final: Annotated[float, Field(ge=0)]
    value_loss_coefficient: Annotated[float, Field(ge=0)]
    max_grad_norm: Annotated[float, Field(gt=0)]
    amp: Literal[False]
    extension_gate_artifact: str | None

    @model_validator(mode="after")
    def validate_rollout_and_training_budget(self) -> PPOConfig:
        rollout_size = self.workers * self.worker_steps
        if rollout_size != 16_384:
            raise ValueError("rollout must contain exactly 16,384 environment steps")
        if rollout_size // self.minibatches != self.effective_minibatch_size:
            raise ValueError("minibatch count does not yield 2,048-sample minibatches")
        if rollout_size % self.minibatches:
            raise ValueError("rollout size must be divisible by minibatch count")
        if self.effective_minibatch_size % self.microbatch_size:
            raise ValueError("effective minibatch must be divisible by microbatch size")
        expected_steps = self.updates * rollout_size
        if self.environment_steps != expected_steps:
            raise ValueError(
                f"environment_steps must equal updates * 16,384 ({expected_steps})"
            )
        if self.updates == 62 and not self.extension_gate_artifact:
            raise ValueError("62 updates require an extension_gate_artifact")
        if self.updates == 31 and self.extension_gate_artifact is not None:
            raise ValueError("extension_gate_artifact is only valid for the 62-update stage")
        return self


class MuonScheduleConfig(StrictModel):
    initial_lr: Literal[0.02]
    final_lr: Literal[0.00067]


class AdamWHeadsScheduleConfig(StrictModel):
    initial_lr: Literal[0.0001]
    final_lr: Literal[0.00000335]


class MuonConfig(StrictModel):
    name: Literal["muon"]
    muon: MuonScheduleConfig
    adamw_heads: AdamWHeadsScheduleConfig
    momentum: Literal[0.95]
    nesterov: Literal[True]
    ns_steps: Literal[5]
    weight_decay: Literal[0.01]
    adjust_lr_fn: Literal["original"]


class DiagnosticsConfig(StrictModel):
    enabled: bool
    detailed_routing_sample_rate: Annotated[float, Field(ge=0, le=1)]
    artifact_name: str
    output_path: str


class CheckpointConfig(StrictModel):
    local_dir: str
    drive_dir: str
    every_updates: Literal[5]
    milestone_updates: list[Literal[31, 62]]
    commit_marker: Literal["commit_success.json"]
    resume_from: str | None

    @model_validator(mode="after")
    def validate_milestones(self) -> CheckpointConfig:
        if 31 not in self.milestone_updates:
            raise ValueError("checkpoint milestone_updates must contain update 31")
        if len(set(self.milestone_updates)) != len(self.milestone_updates):
            raise ValueError("checkpoint milestone_updates must not contain duplicates")
        return self


class WandbConfig(StrictModel):
    enabled: bool
    entity: str | None
    project: str
    run_name: str
    mode: Literal["online", "offline", "disabled"]


class DriveConfig(StrictModel):
    enabled: bool
    root: str


class TrainSeeds(StrictModel):
    model: Literal[1, 2, 3]
    environment_start: Annotated[int, Field(ge=0)]
    environment_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_training_pool(self) -> TrainSeeds:
        last_seed = self.environment_start + self.environment_count - 1
        if last_seed > 9_999:
            raise ValueError("training environment seeds must be within 0..9999")
        return self


class EvaluationSeeds(StrictModel):
    environment_start: Literal[10000]
    environment_count: Literal[50]
    action_rng_repeats: Literal[2, 3]


class MiniGridTransferGateConfig(StrictModel):
    """Untouched-checkpoint S9 transfer measurement and optional threshold gate."""

    environment_start: Literal[10000]
    environment_count: Literal[50]
    action_rng_repeats: Literal[3]
    minimum_success_rate: Literal[0.9]
    minimum_mean_return: Literal[0.8]
    enforce_thresholds: bool = True
    output_path: str


class MiniGridEvaluationProtocol(StrictModel):
    """Fixed final-only, per-delay evaluation protocol."""

    delay_conditions: list[Literal[32, 64, 96, 128]]
    seeds: EvaluationSeeds
    checkpoint_update: Literal[31]
    output_path: str
    summary_path: str

    @model_validator(mode="after")
    def validate_delay_sweep(self) -> MiniGridEvaluationProtocol:
        if self.delay_conditions != [32, 64, 96, 128]:
            raise ValueError("MiniGrid evaluation delays must be [32, 64, 96, 128]")
        if self.seeds.action_rng_repeats != 3:
            raise ValueError("MiniGrid final evaluation requires three paired action-RNG repeats")
        return self


class MiniGridWandbConfig(WandbConfig):
    group: Literal["minigrid-delay-sweep"]


class TrainConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["train"]
    arm: Arm
    provenance: ProvenanceConfig
    environment: EnvironmentConfig
    transformer: TransformerConfig
    lora: LoraConfig
    attention: AttentionConfig
    ppo: PPOConfig
    optimizer: MuonConfig
    diagnostics: DiagnosticsConfig
    checkpointing: CheckpointConfig
    seeds: TrainSeeds
    wandb: WandbConfig
    drive: DriveConfig

    @model_validator(mode="after")
    def validate_experimental_contract(self) -> TrainConfig:
        if self.environment.command_count != 40:
            raise ValueError("training command_count must be 40")
        if self.environment.explosion_delay != 5:
            raise ValueError(
                "training must preserve checkpoint explosion_delay 5"
            )
        if self.environment.seed_start != self.seeds.environment_start:
            raise ValueError("environment seed_start must match seeds.environment_start")
        if self.environment.seed_count != self.seeds.environment_count:
            raise ValueError("environment seed_count must match seeds.environment_count")
        if not self.lora.enabled:
            raise ValueError("LoRA must be enabled for PPO fine-tuning")
        if self.arm == "trxl":
            if self.attention.retrieval.enabled:
                raise ValueError("trxl must disable retrieval")
            if self.attention.dense_recent != self.attention.budget:
                raise ValueError("trxl dense_recent must equal its attention budget")
            if self.attention.retrieval.retrieved_blocks != 0:
                raise ValueError("trxl must retrieve zero blocks")
        else:
            if not self.attention.retrieval.enabled:
                raise ValueError("trxl_moba must enable retrieval")
            if self.attention.budget != 256 or self.attention.dense_recent != 128:
                raise ValueError("primary trxl_moba arm requires budget 256 and dense_recent 128")
            if self.attention.retrieval.block_size != 16:
                raise ValueError("primary trxl_moba arm requires block_size 16")
            if self.attention.retrieval.retrieved_blocks != 8:
                raise ValueError("primary trxl_moba arm requires eight retrieved blocks")
            if self.attention.search_horizon != 2560:
                raise ValueError("primary trxl_moba arm requires search_horizon 2560")
        if self.attention.budget != 256:
            raise ValueError("both fine-tuning arms require an attention budget of 256")
        terminal_update = self.ppo.updates
        if terminal_update not in self.checkpointing.milestone_updates:
            raise ValueError("checkpoint milestones must include the terminal PPO update")
        return self


class MiniGridTrainConfig(StrictModel):
    """Typed, final-only MiniGrid delay-sweep training experiment.

    It intentionally does not inherit from ``TrainConfig``: Mortar's 118-token,
    pre-layer-norm and command-count constraints are historical contracts rather
    than defaults for a different pretrained checkpoint.
    """

    schema_version: Literal[1]
    task: Literal["train-minigrid"]
    arm: Arm
    provenance: ProvenanceConfig
    environment: MiniGridEnvironmentConfig
    transformer: MiniGridTransformerConfig
    lora: LoraConfig
    attention: AttentionConfig
    ppo: PPOConfig
    optimizer: MuonConfig
    diagnostics: DiagnosticsConfig
    checkpointing: CheckpointConfig
    seeds: TrainSeeds
    wandb: MiniGridWandbConfig
    drive: DriveConfig
    transfer_gate: MiniGridTransferGateConfig
    evaluation: MiniGridEvaluationProtocol

    @model_validator(mode="after")
    def validate_experimental_contract(self) -> MiniGridTrainConfig:
        if self.provenance.checkpoint_path != "models/minigrid.nn":
            raise ValueError("MiniGrid must start from models/minigrid.nn")
        if self.provenance.checkpoint_sha256 != (
            "11065c3fb00abe08555ff4ddb436285cf3351920b5ad1ba8978e071f00dc1375"
        ):
            raise ValueError("MiniGrid config must use the pinned minigrid checkpoint SHA-256")
        if self.ppo.updates != 31 or self.ppo.environment_steps != 507_904:
            raise ValueError("MiniGrid delay sweep is locked to update 31 / 507,904 steps")
        if self.environment.seed_start != self.seeds.environment_start:
            raise ValueError("environment seed_start must match seeds.environment_start")
        if self.environment.seed_count != self.seeds.environment_count:
            raise ValueError("environment seed_count must match seeds.environment_count")
        if not self.lora.enabled:
            raise ValueError("LoRA must be enabled for MiniGrid PPO fine-tuning")
        if self.arm == "trxl":
            if self.attention.retrieval.enabled:
                raise ValueError("trxl must disable retrieval")
            if self.attention.dense_recent != self.attention.budget:
                raise ValueError("trxl dense_recent must equal its attention budget")
            if self.attention.retrieval.retrieved_blocks != 0:
                raise ValueError("trxl must retrieve zero blocks")
        else:
            if not self.attention.retrieval.enabled:
                raise ValueError("trxl_moba must enable retrieval")
            if self.attention.budget != 256 or self.attention.dense_recent != 128:
                raise ValueError("trxl_moba requires budget 256 and dense_recent 128")
            if self.attention.retrieval.block_size != 16:
                raise ValueError("trxl_moba requires block_size 16")
            if self.attention.retrieval.retrieved_blocks != 8:
                raise ValueError("trxl_moba requires eight retrieved blocks")
            if self.attention.search_horizon != 2560:
                raise ValueError("trxl_moba requires search_horizon 2560")
        if self.attention.budget != 256:
            raise ValueError("both MiniGrid arms require an attention budget of 256")
        if self.ppo.updates not in self.checkpointing.milestone_updates:
            raise ValueError("checkpoint milestones must include update 31")
        if self.evaluation.checkpoint_update not in self.checkpointing.milestone_updates:
            raise ValueError("final MiniGrid evaluation must select a checkpointed update")
        return self


class EvaluationProtocol(StrictModel):
    command_counts: list[int]
    seeds: EvaluationSeeds
    checkpoint_path: str
    output_path: str

    @model_validator(mode="after")
    def validate_primary_protocol(self) -> EvaluationProtocol:
        if self.command_counts != PRIMARY_COMMAND_COUNTS:
            raise ValueError(
                "primary evaluation command_counts must be [10, 20, 30, 40, 50, 60, 80]"
            )
        if self.seeds.action_rng_repeats != 3:
            raise ValueError("primary evaluation requires three paired action-RNG repeats")
        return self


class EvaluateConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["evaluate"]
    arm: Arm
    provenance: ProvenanceConfig
    evaluation: EvaluationProtocol


class BaselineGateConfig(StrictModel):
    command_count: Literal[10]
    model_max_episode_steps: Literal[119]
    seeds: EvaluationSeeds
    minimum_success_rate: Literal[0.95]
    minimum_mean_normalized_return: Literal[0.95]
    output_path: str

    @model_validator(mode="after")
    def validate_repeats(self) -> BaselineGateConfig:
        if self.seeds.action_rng_repeats != 2:
            raise ValueError("pretrained baseline requires two paired action-RNG repeats")
        return self


class PretrainedEvalConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["eval-pretrained"]
    arm: Literal["trxl"]
    provenance: ProvenanceConfig
    environment: EnvironmentConfig
    transformer: TransformerConfig
    lora: LoraConfig
    attention: AttentionConfig
    evaluation: BaselineGateConfig

    @model_validator(mode="after")
    def validate_untouched_baseline(self) -> PretrainedEvalConfig:
        if self.lora.enabled:
            raise ValueError("pretrained baseline must not attach LoRA")
        if self.environment.command_count != 10:
            raise ValueError("pretrained baseline environment command_count must be 10")
        if self.environment.seed_start != 10000 or self.environment.seed_count != 50:
            raise ValueError("pretrained baseline must use held-out seeds 10000..10049")
        if self.environment.explosion_delay != 5:
            raise ValueError(
                "pretrained baseline must preserve checkpoint explosion_delay 5"
            )
        if self.attention.retrieval.enabled:
            raise ValueError("pretrained baseline must disable retrieval")
        if self.attention.budget != 118 or self.attention.dense_recent != 118:
            raise ValueError("pretrained baseline must preserve the checkpoint's 118-memory context")
        return self


class AnalyzeRetrievalConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["analyze-retrieval"]
    arm: Literal["trxl_moba"]
    resolved_config_path: str
    routing_artifact_path: str
    output_csv: str
    output_dir: str


RunConfig: TypeAlias = Annotated[
    (
        TrainConfig
        | MiniGridTrainConfig
        | EvaluateConfig
        | PretrainedEvalConfig
        | AnalyzeRetrievalConfig
    ),
    Field(discriminator="task"),
]
_RUN_CONFIG_ADAPTER = TypeAdapter(RunConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load one YAML document and validate it against its task-specific schema."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration must contain exactly one YAML mapping")
    return _RUN_CONFIG_ADAPTER.validate_python(raw)
