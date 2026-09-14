import json
import pickle

import pytest

from episodic_moba_ppo.checkpoint import (
    MARKER_NAME,
    PAYLOAD_NAME,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointStore,
    load_legacy_checkpoint,
    sha256_file,
    verify_file_hash,
)


def dump_pickle(payload, path):
    with open(path, "wb") as stream:
        pickle.dump(payload, stream)


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def test_hash_verification_and_legacy_load(tmp_path):
    path = tmp_path / "legacy.nn"
    dump_pickle(({"weight": 1}, {"transformer": {}}), path)
    digest = sha256_file(path)
    state, config = load_legacy_checkpoint(path, digest)
    assert state == {"weight": 1}
    assert config == {"transformer": {}}
    with pytest.raises(CheckpointIntegrityError):
        verify_file_hash(path, "0" * 64)


def test_commit_writes_marker_last_and_round_trips(tmp_path):
    events = []
    store = CheckpointStore(
        tmp_path, dump=dump_pickle, load=load_pickle, fault_hook=events.append
    )
    directory = store.commit(5, {"step": 81_920}, {"arm": "trxl", "seed": 1})
    assert events == ["payload_committed", "before_marker", "marker_committed"]
    assert (directory / PAYLOAD_NAME).is_file()
    assert (directory / MARKER_NAME).is_file()
    recovered = store.recover_latest()
    assert recovered.update == 5
    assert recovered.payload == {"step": 81_920}


def test_interrupted_commit_without_marker_is_not_recoverable(tmp_path):
    def fail_before_marker(stage):
        if stage == "before_marker":
            raise RuntimeError("simulated Drive interruption")

    store = CheckpointStore(
        tmp_path, dump=dump_pickle, load=load_pickle, fault_hook=fail_before_marker
    )
    with pytest.raises(RuntimeError):
        store.commit(1, {"step": 1}, {})
    assert (tmp_path / "update-00001" / PAYLOAD_NAME).is_file()
    assert not (tmp_path / "update-00001" / MARKER_NAME).exists()
    with pytest.raises(CheckpointError):
        store.recover_latest()

    retry = CheckpointStore(tmp_path, dump=dump_pickle, load=load_pickle)
    retry.commit(1, {"step": 2}, {})
    assert retry.recover_latest().payload == {"step": 2}


def test_recovery_skips_corrupt_newest_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path, dump=dump_pickle, load=load_pickle)
    store.commit(1, {"step": 1}, {})
    newest = store.commit(2, {"step": 2}, {})
    with open(newest / PAYLOAD_NAME, "ab") as stream:
        stream.write(b"corrupt")
    recovered = store.recover_latest()
    assert recovered.update == 1


def test_recovery_skips_hash_valid_but_unloadable_newest(tmp_path):
    store = CheckpointStore(tmp_path, dump=dump_pickle, load=load_pickle)
    store.commit(1, {"step": 1}, {})
    newest = store.commit(2, {"step": 2}, {})
    (newest / PAYLOAD_NAME).write_bytes(b"not a pickle")
    marker_path = newest / MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker["payloads"][0]["size"] = (newest / PAYLOAD_NAME).stat().st_size
    marker["payloads"][0]["sha256"] = sha256_file(newest / PAYLOAD_NAME)
    marker_path.write_text(json.dumps(marker))
    recovered = store.recover_latest()
    assert recovered.update == 1
