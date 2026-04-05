from pathlib import Path

from the_oracle.platform_support import invalid_windows_path_parts


def test_repository_paths_are_windows_portable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in repo_root.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(repo_root)
        invalid = invalid_windows_path_parts(relative)
        if invalid:
            offenders.append(relative.as_posix())
    assert offenders == []
