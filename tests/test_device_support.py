from the_oracle.device_support import available_device_modes, resolve_chatterbox_device


def test_cpu_device_mode_is_available() -> None:
    modes = {item.key: item for item in available_device_modes()}

    assert modes["cpu"].available is True
    assert resolve_chatterbox_device("cpu") == "cpu"


def test_vulkan_mode_fails_clearly_when_unverified() -> None:
    try:
        resolve_chatterbox_device("vulkan")
    except RuntimeError as exc:
        assert "not verified" in str(exc)
    else:
        raise AssertionError("Expected Vulkan mode to fail clearly in the current runtime")
