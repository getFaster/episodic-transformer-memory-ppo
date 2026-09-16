#!/usr/bin/env bash
set -Eeuo pipefail

# Run the 507,904-step pilot for model seeds 1, 2, and 3 concurrently.
# Usage:
#   DRIVE_ROOT=/content/drive/MyDrive/episodic-moba-ppo \
#     bash scripts/run_three_seeds.sh trxl_moba
#   DRIVE_ROOT=/content/drive/MyDrive/episodic-moba-ppo \
#     bash scripts/run_three_seeds.sh trxl_moba --memory-limit 18G

arm="trxl_moba"
memory_limit=""
internal_seed=""
arm_seen=0
while (($#)); do
  case "$1" in
    trxl|trxl_moba)
      if ((arm_seen)); then
        echo "arm may be specified only once" >&2
        exit 2
      fi
      arm="$1"
      arm_seen=1
      shift
      ;;
    --memory-limit)
      if (($# < 2)) || [[ -z "$2" ]]; then
        echo "--memory-limit requires a value such as 18G" >&2
        exit 2
      fi
      memory_limit="$2"
      shift 2
      ;;
    --run-seed)
      if (($# < 2)) || [[ ! "$2" =~ ^[123]$ ]]; then
        echo "--run-seed requires 1, 2, or 3" >&2
        exit 2
      fi
      internal_seed="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [trxl|trxl_moba] [--memory-limit SIZE]" >&2
      exit 2
      ;;
  esac
done

: "${DRIVE_ROOT:?Set DRIVE_ROOT to the mounted experiment directory}"
wandb_entity="${WANDB_ENTITY:-}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
work_root="${WORK_ROOT:-/tmp/episodic-moba-ppo-runs}"
baseline_reference="${work_root}/baseline_reference.json"
base_config="${repo_root}/configs/${arm}_command40.yaml"

if [[ -n "${memory_limit}" && -z "${internal_seed}" && "${EPISODIC_MEMORY_GUARD_ACTIVE:-}" != "1" ]]; then
  if ! command -v systemd-run >/dev/null || ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "--memory-limit requires a working systemd user manager" >&2
    exit 2
  fi
  if ! numfmt --from=iec "${memory_limit}" >/dev/null 2>&1; then
    echo "invalid --memory-limit value: ${memory_limit}" >&2
    exit 2
  fi
  memory_limit_bytes="$(numfmt --from=iec "${memory_limit}")"
  if ((memory_limit_bytes <= 512 * 1024 * 1024)); then
    echo "--memory-limit must exceed 512 MiB" >&2
    exit 2
  fi
  memory_high_bytes="$((memory_limit_bytes - 256 * 1024 * 1024))"
  unit="episodic-moba-three-seeds-${BASHPID}"
  exec systemd-run --user --scope --collect --quiet --expand-environment=no \
    --unit="${unit}" \
    --property=MemoryAccounting=yes \
    --property="MemoryHigh=${memory_high_bytes}" \
    --property="MemoryMax=${memory_limit}" \
    --property=MemorySwapMax=0 \
    --property=OOMPolicy=continue \
    --property=Delegate=yes \
    /usr/bin/env EPISODIC_MEMORY_GUARD_ACTIVE=1 \
    bash "${script_dir}/run_three_seeds.sh" "${arm}" \
      --memory-limit "${memory_limit}"
fi

if [[ -z "${internal_seed}" ]]; then
  mkdir -p "${work_root}/configs" "${work_root}/logs"
  mkdir -p "${DRIVE_ROOT}/evaluations" "${DRIVE_ROOT}/analysis"

  cd "${repo_root}"
  uv sync --frozen --python 3.11
  uv run eval-pretrained \
    --config configs/pretrained_eval.yaml \
    --repo-root "${repo_root}" \
    --output "${baseline_reference}"
fi

run_seed() {
  local seed="$1"
  local run_id="${arm}-command40-seed${seed}"
  local resolved_config="${work_root}/configs/${run_id}.yaml"
  local routing_csv="${DRIVE_ROOT}/analysis/${run_id}/routing-details.csv"
  local drive_run_dir="${DRIVE_ROOT}/checkpoints/${run_id}"
  local progress_path="${work_root}/progress/${run_id}.json"

  local resume_from=""
  if [[ -d "${drive_run_dir}" ]]; then
    resume_from="$(
      find "${drive_run_dir}" -mindepth 2 -maxdepth 2 \
        -type f -name commit_success.json -printf '%h\n' \
        | sort | tail -n 1
    )"
  fi

  uv run python - \
    "${base_config}" "${resolved_config}" "${arm}" "${seed}" \
    "${run_id}" "${wandb_entity}" "${DRIVE_ROOT}" "${work_root}" \
    "${routing_csv}" "${resume_from}" <<'PY'
from pathlib import Path
import sys

import yaml

(
    base_config,
    resolved_config,
    arm,
    seed_text,
    run_id,
    wandb_entity,
    drive_root,
    work_root,
    routing_csv,
    resume_from,
) = sys.argv[1:]

with Path(base_config).open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

config["seeds"]["model"] = int(seed_text)
config["wandb"]["entity"] = wandb_entity or None
config["wandb"]["run_name"] = run_id
config["wandb"]["run_id"] = run_id
config["drive"]["root"] = drive_root
config["checkpointing"]["local_dir"] = str(Path(work_root) / "checkpoints" / run_id)
config["checkpointing"]["drive_dir"] = f"checkpoints/{run_id}"
config["checkpointing"]["resume_from"] = resume_from or None
config["diagnostics"]["output_path"] = routing_csv

with Path(resolved_config).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

  echo "Starting ${run_id}${resume_from:+ from ${resume_from}}"
  train_args=(
    --config "${resolved_config}"
    --baseline-reference "${baseline_reference}"
    --repo-root "${repo_root}"
  )
  if [[ -n "${memory_limit}" ]]; then
    train_args+=(--progress-path "${progress_path}")
  fi
  PYTHONHASHSEED="${seed}" uv run train "${train_args[@]}"

  checkpoint_dir="${drive_run_dir}/update-00031"
  if [[ ! -f "${checkpoint_dir}/commit_success.json" ]]; then
    echo "Missing durable update-31 checkpoint: ${checkpoint_dir}" >&2
    exit 1
  fi

  uv run evaluate \
    --checkpoint "${checkpoint_dir}" \
    --arm "${arm}" \
    --model-seed "${seed}" \
    --output "${DRIVE_ROOT}/evaluations/${run_id}.json" \
    --repo-root "${repo_root}"

  if [[ "${arm}" == "trxl_moba" ]]; then
    analysis_config="${work_root}/configs/analyze-${run_id}.yaml"
    analysis_dir="${DRIVE_ROOT}/analysis/${run_id}"
    uv run python - \
      "${resolved_config}" "${routing_csv}" "${analysis_config}" \
      "${analysis_dir}" <<'PY'
from pathlib import Path
import sys

import yaml

resolved_config, routing_csv, analysis_config, analysis_dir = sys.argv[1:]
document = {
    "schema_version": 1,
    "task": "analyze-retrieval",
    "arm": "trxl_moba",
    "resolved_config_path": resolved_config,
    "routing_artifact_path": routing_csv,
    "output_csv": str(Path(analysis_dir) / "routing-summary.csv"),
    "output_dir": analysis_dir,
}
Path(analysis_dir).mkdir(parents=True, exist_ok=True)
with Path(analysis_config).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(document, stream, sort_keys=False)
PY
    uv run analyze-retrieval --config "${analysis_config}"
  fi
}

if [[ -n "${internal_seed}" ]]; then
  run_seed "${internal_seed}"
  exit $?
fi

mkdir -p "${work_root}/progress"

pids=()
seeds=()
progress_paths=()
cgroup_children=()
guard_pid=""

cleanup() {
  local status="$?"
  trap - EXIT
  if [[ -n "${guard_pid}" ]]; then
    kill "${guard_pid}" 2>/dev/null || true
  fi
  if [[ -n "${memory_limit}" ]]; then
    for pid in "${pids[@]}"; do
      [[ -n "${pid}" ]] || continue
      kill -CONT -- "-${pid}" 2>/dev/null || true
      kill -TERM -- "-${pid}" 2>/dev/null || true
    done
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

memory_guard_enabled=0
if [[ -n "${memory_limit}" ]]; then
  memory_guard_enabled=1
  cgroup_relative="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
  cgroup_root="/sys/fs/cgroup${cgroup_relative}"
  if [[ ! -w "${cgroup_root}/cgroup.procs" || ! -w "${cgroup_root}/cgroup.subtree_control" ]]; then
    echo "delegated cgroup is not writable: ${cgroup_root}" >&2
    exit 2
  fi
  supervisor_cgroup="${cgroup_root}/supervisor"
  mkdir "${supervisor_cgroup}"
  echo "${BASHPID}" > "${supervisor_cgroup}/cgroup.procs"
  echo +memory > "${cgroup_root}/cgroup.subtree_control"
fi

for seed in 1 2 3; do
  log_path="${work_root}/logs/${arm}-command40-seed${seed}.log"
  progress_path="${work_root}/progress/${arm}-command40-seed${seed}.json"
  rm -f "${progress_path}"
  echo "Launching ${arm} seed ${seed}; log: ${log_path}"
  if ((memory_guard_enabled)); then
    child_cgroup="${cgroup_root}/seed${seed}"
    mkdir "${child_cgroup}"
    echo 1 > "${child_cgroup}/memory.oom.group"
    setsid bash -c \
      'kill -STOP "$$"; exec bash "$1" "$2" --memory-limit "$3" --run-seed "$4"' \
      _ "${script_dir}/run_three_seeds.sh" "${arm}" "${memory_limit}" "${seed}" \
      > >(tee "${log_path}") 2>&1 &
    pid="$!"
    for _ in {1..100}; do
      state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
      [[ "${state}" == "T" ]] && break
      sleep 0.01
    done
    if [[ "${state:-}" != "T" ]]; then
      echo "seed ${seed} did not enter its cgroup launch barrier" >&2
      kill "${pid}" 2>/dev/null || true
      exit 2
    fi
    echo "${pid}" > "${child_cgroup}/cgroup.procs"
    echo "$(((seed - 1) * 500))" > "/proc/${pid}/oom_score_adj"
    kill -CONT "${pid}"
    cgroup_children+=("${child_cgroup}")
  else
    run_seed "${seed}" > >(tee "${log_path}") 2>&1 &
    pid="$!"
  fi
  pids+=("${pid}")
  seeds+=("${seed}")
  progress_paths+=("${progress_path}")
done

if ((memory_guard_enabled)); then
  guard_args=()
  for index in "${!pids[@]}"; do
    guard_args+=(
      --seed-process
      "${seeds[$index]}:${pids[$index]}:${progress_paths[$index]}:${cgroup_children[$index]}"
    )
  done
  uv run python -m episodic_moba_ppo.memory_guard \
    --cgroup "${cgroup_root}" \
    --memory-limit "${memory_limit}" \
    "${guard_args[@]}" &
  guard_pid="$!"
fi

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${arm} seed ${seeds[$index]}"
  else
    status=$?
    echo "Failed ${arm} seed ${seeds[$index]} with status ${status}" >&2
    failed=1
  fi
  pids[$index]=""
done

if [[ -n "${guard_pid}" ]]; then
  wait "${guard_pid}" || failed=1
  guard_pid=""
fi

for child_cgroup in "${cgroup_children[@]}"; do
  rmdir "${child_cgroup}" 2>/dev/null || true
done

exit "${failed}"
