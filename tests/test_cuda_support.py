"""Focused tests for CUDA capability detection and PyTorch device selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import the_oracle.device_support as device_support
from the_oracle.cli import build_parser
from the_oracle.device_support import CUDA_MIN_VRAM_BYTES, CUDADeviceInfo


def _fake_torch(*, available: bool, devices: list[tuple[str, int]]):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(devices) if available else 0,
        get_device_properties=lambda index: SimpleNamespace(
            name=devices[index][0], total_memory=devices[index][1]
        ),
    )
    return SimpleNamespace(cuda=cuda)


def test_cuda_devices_reports_suitable_and_undersized_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _fake_torch(
        available=True,
        devices=[
            ("Legacy 1GB", 1 * 1024**3),
            ("Workstation 8GB", 8 * 1024**3),
        ],
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    devices = device_support._torch_cuda_devices()

    assert [device.name for device in devices] == ["Legacy 1GB", "Workstation 8GB"]
    assert devices[0].suitable is False
    assert "below the 4 GiB minimum" in devices[0].reason
    assert devices[1].suitable is True
    assert devices[1].torch_available is True
    assert devices[1].label == "CUDA 1: Workstation 8GB (8.0 GiB VRAM)"


def test_cuda_resolution_defaults_to_cuda_and_can_pin_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [
        CUDADeviceInfo(
            index=0,
            name="Good GPU",
            vram_bytes=8 * 1024**3,
            torch_available=True,
            suitable=True,
        )
    ]
    monkeypatch.setattr(device_support, "cuda_devices", lambda: devices)

    assert device_support.resolve_chatterbox_device("cuda") == "cuda"
    assert device_support.resolve_chatterbox_device("cuda", 0) == "cuda:0"


def test_cuda_resolution_rejects_missing_runtime_and_bad_index(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [
        CUDADeviceInfo(
            index=0,
            name="Legacy GPU",
            vram_bytes=1 * 1024**3,
            torch_available=False,
            suitable=False,
            reason="too small",
        )
    ]
    monkeypatch.setattr(device_support, "cuda_devices", lambda: devices)

    with pytest.raises(RuntimeError, match="CUDA inference is unavailable"):
        device_support.resolve_chatterbox_device("cuda")

    devices[0] = CUDADeviceInfo(
        index=0,
        name="Good GPU",
        vram_bytes=8 * 1024**3,
        torch_available=True,
        suitable=True,
    )
    with pytest.raises(RuntimeError, match="CUDA device 3"):
        device_support.resolve_chatterbox_device("cuda", 3)


def test_nvidia_smi_fallback_classifies_legacy_card(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="0, Old NVIDIA, 1024, 390.157\n1, New NVIDIA, 8192, 550.54\n",
    )
    monkeypatch.setattr(device_support.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(device_support.subprocess, "run", lambda *args, **kwargs: completed)

    devices = device_support._nvidia_smi_devices()

    assert devices[0].vram_bytes == 1024 * 1024**2
    assert devices[0].suitable is False
    assert devices[1].vram_bytes == 8192 * 1024**2
    assert devices[1].suitable is True
    assert devices[1].driver_version == "550.54"


def test_available_cuda_mode_is_runtime_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_support, "cuda_devices", lambda: [])
    modes = {mode.key: mode for mode in device_support.available_device_modes()}

    assert modes["cpu"].available is True
    assert modes["cuda"].available is False
    assert "No NVIDIA" in modes["cuda"].reason


def test_cuda_minimum_is_four_gib() -> None:
    assert CUDA_MIN_VRAM_BYTES == 4 * 1024**3


def test_cli_parser_accepts_cuda_device_selection() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "render",
        "--input", "input.txt",
        "--outdir", "Output",
        "--speakerA-ref", "a.wav",
        "--speakerB-ref", "b.wav",
        "--device-mode", "cuda",
        "--cuda-device", "1",
    ])

    assert args.device_mode == "cuda"
    assert args.cuda_device == 1


def test_cli_parser_rejects_negative_cuda_device() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "render",
            "--device-mode", "cuda",
            "--cuda-device", "-1",
        ])
