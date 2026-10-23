from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_numpy_constraint_matches_chatterbox_017_supported_python_range() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    numpy = next(
        Requirement(item)
        for item in metadata["project"]["dependencies"]
        if Requirement(item).name.lower() == "numpy"
    )

    assert numpy.specifier.contains(Version("1.26.4"))
    assert not numpy.specifier.contains(Version("2.4.6"))
