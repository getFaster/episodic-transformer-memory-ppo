import pickle
from pathlib import Path

from model import ActorCriticModel


class _ObservationSpace:
    shape = (3, 84, 84)


def test_shipped_mortar_checkpoint_loads_strictly():
    checkpoint_path = (
        Path(__file__).parents[1] / "models" / "mortar_mayhem_grid_trxl.nn"
    )
    with checkpoint_path.open("rb") as checkpoint_file:
        state_dict, config = pickle.load(checkpoint_file)

    model = ActorCriticModel(
        config=config,
        observation_space=_ObservationSpace(),
        action_space_shape=(4,),
        max_episode_length=512,
    )
    incompatible = model.load_state_dict(state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for layer_index in range(3):
        prefix = f"transformer.transformer_blocks.{layer_index}.attention"
        assert tuple(state_dict[f"{prefix}.queries.weight"].shape) == (96, 96)
        assert tuple(state_dict[f"{prefix}.keys.weight"].shape) == (96, 96)
        assert tuple(state_dict[f"{prefix}.values.weight"].shape) == (96, 96)
        assert tuple(state_dict[f"{prefix}.fc_out.weight"].shape) == (384, 384)
