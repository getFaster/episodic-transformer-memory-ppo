import tomllib
from pathlib import Path


def test_peft_is_pinned_exactly_in_project_and_lockfile() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as project_file:
        dependencies = tomllib.load(project_file)["project"]["dependencies"]

    assert "peft==0.20.0" in dependencies
    lockfile = (root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "peft"\nversion = "0.20.0"' in lockfile
    assert '{ name = "peft", specifier = "==0.20.0" }' in lockfile
