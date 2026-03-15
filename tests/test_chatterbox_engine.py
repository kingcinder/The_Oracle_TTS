import sys
from pathlib import Path
from types import SimpleNamespace

from the_oracle.tts_engines import chatterbox_engine


def test_download_turbo_checkpoint_does_not_force_true_token(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / "turbo-cache")

    monkeypatch.setattr(chatterbox_engine, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    checkpoint_dir = chatterbox_engine.download_turbo_checkpoint()

    assert checkpoint_dir == tmp_path / "turbo-cache"
    assert calls["repo_id"] == chatterbox_engine.TURBO_REPO_ID
    assert calls["token"] is None
    assert calls["allow_patterns"] == chatterbox_engine.TURBO_ALLOW_PATTERNS


def test_turbo_engine_loads_from_local_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "turbo-cache"
    checkpoint_dir.mkdir()
    calls: list[tuple[Path, str]] = []

    class FakeTurboModel:
        sr = 24000

    class FakeTurboTTS:
        @classmethod
        def from_local(cls, ckpt_dir, device):
            calls.append((Path(ckpt_dir), device))
            return FakeTurboModel()

    monkeypatch.setattr(chatterbox_engine, "download_turbo_checkpoint", lambda local_files_only=False: checkpoint_dir)
    monkeypatch.setitem(
        sys.modules,
        "chatterbox.tts_turbo",
        SimpleNamespace(ChatterboxTurboTTS=FakeTurboTTS, Conditionals=object),
    )

    engine = chatterbox_engine.ChatterboxEngine(variant="turbo", device="cpu")
    model, condition_cls, languages = engine._load_variant()

    assert isinstance(model, FakeTurboModel)
    assert condition_cls is object
    assert languages == {"en": "English"}
    assert calls == [(checkpoint_dir, "cpu")]


def test_turbo_readiness_report_explains_broken_auth_loader(monkeypatch) -> None:
    monkeypatch.setattr(
        chatterbox_engine,
        "download_turbo_checkpoint",
        lambda local_files_only=False: (_ for _ in ()).throw(RuntimeError("Token is required (`token=True`), but no token found.")),
    )

    report = chatterbox_engine.turbo_readiness_report(device="cpu")

    assert report["ok"] is False
    assert report["cached"] is False
    assert "forces HF auth" in report["error"]
