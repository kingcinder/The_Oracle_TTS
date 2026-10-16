from pathlib import Path

from the_oracle.app_paths import (
    PORTABLE_MARKER,
    default_output_filename,
    ensure_repo_default_paths,
    normalize_output_filename,
    resolve_output_filename,
)


def test_portable_marker_keeps_repo_local_paths(tmp_path: Path) -> None:
    (tmp_path / PORTABLE_MARKER).touch()
    paths = ensure_repo_default_paths(tmp_path)
    assert paths.input_dir == tmp_path / "Input"
    assert paths.output_dir == tmp_path / "Output"


def test_installed_root_migrates_repo_content_without_deleting_it(tmp_path: Path, monkeypatch) -> None:
    import the_oracle.app_paths as app_paths

    repo = tmp_path / "repo"
    (repo / "Input").mkdir(parents=True)
    (repo / "Input" / "story.txt").write_text("A: hello\n", encoding="utf-8")
    monkeypatch.setattr(app_paths, "__file__", str(repo / "src" / "the_oracle" / "app_paths.py"))
    monkeypatch.setattr(app_paths, "app_data_dir", lambda: tmp_path / "data" / "the_oracle")

    paths = app_paths.ensure_repo_default_paths(repo)

    assert paths.input_dir == tmp_path / "data" / "the_oracle" / "Input"
    assert (paths.input_dir / "story.txt").read_text(encoding="utf-8") == "A: hello\n"
    assert (repo / "Input" / "story.txt").is_file()


def test_ensure_repo_default_paths_creates_expected_directories(tmp_path: Path) -> None:
    paths = ensure_repo_default_paths(tmp_path)

    assert paths.input_dir == tmp_path / "Input"
    assert paths.output_dir == tmp_path / "Output"
    assert paths.profile_dir == tmp_path / "Profiles"
    assert paths.voice_dir == tmp_path / "Seashells"
    assert paths.input_dir.is_dir()
    assert paths.output_dir.is_dir()
    assert paths.profile_dir.is_dir()
    assert paths.voice_dir.is_dir()


def test_output_filename_resolution_uses_default_input_stem_only_for_default_output_dir(tmp_path: Path) -> None:
    default_output_dir = tmp_path / "Output"
    custom_output_dir = tmp_path / "Elsewhere"
    input_path = tmp_path / "story.md"

    assert normalize_output_filename("chapter_one") == "chapter_one.flac"
    assert default_output_filename(input_path) == "story.flac"
    assert resolve_output_filename(input_path, default_output_dir, default_output_dir, "") == "story.flac"
    assert resolve_output_filename(input_path, custom_output_dir, default_output_dir, "") == ""
    assert resolve_output_filename(input_path, custom_output_dir, default_output_dir, "custom_name") == "custom_name.flac"
