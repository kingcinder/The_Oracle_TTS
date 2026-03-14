import json
from pathlib import Path

import soundfile as sf

from dualvoice_studio.smoke import run_deterministic_smoke_render


def test_deterministic_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="txt")

    assert result.output_path.exists()
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True

    audio, sample_rate = sf.read(result.output_path, always_2d=False)
    assert sample_rate == 24000
    assert len(audio) > 1000

    render_plan = json.loads(result.render_plan_path.read_text(encoding="utf-8"))
    assert render_plan["engine"] == "chatterbox"
    assert render_plan["metadata"]["model_variant"] == "standard"
    assert render_plan["metadata"]["cache_reused_on_second_pass"] == "True"
    assert render_plan["metadata"]["watermark"] == "Perth watermark embedded by Chatterbox"


def test_deterministic_markdown_smoke_render_runs_end_to_end(tmp_path: Path) -> None:
    result = run_deterministic_smoke_render(tmp_path, source_format="md")

    assert result.output_path.exists()
    assert result.render_plan_path.exists()
    assert result.stem_count == 4
    assert result.cache_reused_on_second_pass is True

    audio, sample_rate = sf.read(result.output_path, always_2d=False)
    assert sample_rate == 24000
    assert len(audio) > 1000

    render_plan = json.loads(result.render_plan_path.read_text(encoding="utf-8"))
    assert render_plan["engine"] == "chatterbox"
    assert render_plan["metadata"]["model_variant"] == "standard"
    assert render_plan["metadata"]["cache_reused_on_second_pass"] == "True"
