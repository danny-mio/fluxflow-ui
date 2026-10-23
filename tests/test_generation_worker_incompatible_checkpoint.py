"""Legacy-decoder checkpoints (IncompatibleCheckpointError) surface a clear error, no fallback chain."""

from unittest.mock import MagicMock, patch

import pytest
from fluxflow.exceptions import IncompatibleCheckpointError

from fluxflow_ui.utils.generation_worker import GenerationWorker

MSG = "Legacy decoder layout (ConvTranspose2d) - migrate or re-train"


@pytest.fixture
def worker():
    w = object.__new__(GenerationWorker)
    w.device = MagicMock()
    w.device.type = "cpu"
    w._load_legacy_model = MagicMock(return_value=(True, "legacy"))
    w._load_v070_fallback = MagicMock(return_value=(True, "v070"))
    return w


@pytest.mark.parametrize("as_dir", [True, False])
def test_incompatible_checkpoint_clear_error(worker, tmp_path, as_dir):
    path = tmp_path / "ckpt"
    path.mkdir() if as_dir else path.write_bytes(b"x")
    with patch(
        "fluxflow_ui.utils.generation_worker.load_versioned_checkpoint",
        side_effect=IncompatibleCheckpointError(MSG),
    ):
        ok, message = worker.load_model(str(path))
    assert not ok
    assert MSG in message
    worker._load_legacy_model.assert_not_called()
    worker._load_v070_fallback.assert_not_called()


def test_v100_fallback_error_passthrough(worker, tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"x")
    worker._load_v100_fallback = MagicMock(side_effect=IncompatibleCheckpointError(MSG))
    with (
        patch(
            "fluxflow_ui.utils.generation_worker.detect_architecture_version", return_value="0.10.0"
        ),
        patch("safetensors.torch.load_file", return_value={}),
    ):
        ok, message = GenerationWorker._load_legacy_model(worker, str(path), 64, 64, 1024, None)
    assert not ok
    assert MSG in message
    assert "save_versioned_checkpoint" not in message
