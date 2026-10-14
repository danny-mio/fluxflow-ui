"""Tests for the opt-in compile_activations option (Pade torch.compile)."""

from unittest.mock import MagicMock, patch

import safetensors.torch
import torch

from fluxflow_ui.utils.config_manager import ConfigManager
from fluxflow_ui.utils.generation_worker import GenerationWorker
from fluxflow_ui.utils.training_runner import TrainingRunner

COMPILE = "fluxflow.models.pade_activation.compile_pade_activations"


def _ckpt(tmp_path, activation):
    path = tmp_path / "model.safetensors"
    meta = {"activation_type": activation} if activation else None
    safetensors.torch.save_file({"w": torch.zeros(2)}, str(path), metadata=meta)
    return str(path)


def _load(tmp_path, activation, **kwargs):
    worker = object.__new__(GenerationWorker)
    with (
        patch(COMPILE) as mock_compile,
        patch(
            "fluxflow_ui.utils.generation_worker.load_versioned_checkpoint",
            side_effect=RuntimeError("stop"),
        ),
        patch.object(GenerationWorker, "_load_legacy_model", return_value=(True, "ok")),
    ):
        worker.load_model(_ckpt(tmp_path, activation), **kwargs)
    return mock_compile


class TestGenerationWorkerCompile:
    def test_pade_enabled_compiles(self, tmp_path):
        _load(tmp_path, "pade", compile_activations=True).assert_called_once_with()

    def test_default_off_does_not_compile(self, tmp_path):
        _load(tmp_path, "pade").assert_not_called()

    def test_bezier_does_not_compile(self, tmp_path):
        _load(tmp_path, "bezier", compile_activations=True).assert_not_called()

    def test_compile_failure_does_not_break_load(self, tmp_path):
        worker = object.__new__(GenerationWorker)
        with (
            patch(COMPILE, side_effect=RuntimeError("boom")),
            patch(
                "fluxflow_ui.utils.generation_worker.load_versioned_checkpoint",
                side_effect=RuntimeError("stop"),
            ),
            patch.object(GenerationWorker, "_load_legacy_model", return_value=(True, "ok")),
        ):
            ok, _ = worker.load_model(_ckpt(tmp_path, "pade"), compile_activations=True)
        assert ok


class TestTrainingRunnerCompile:
    def _cmd(self, tmp_path, **extra):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(stdout=iter([]))
            TrainingRunner().start_training(
                {"data_path": "/d", "output_path": str(tmp_path), **extra}
            )
        return mock_popen.call_args[0][0]

    def test_flag_forwarded_when_enabled(self, tmp_path):
        assert "--compile_activations" in self._cmd(tmp_path, compile_activations=True)

    def test_flag_absent_by_default(self, tmp_path):
        assert "--compile_activations" not in self._cmd(tmp_path)
        assert "--compile_activations" not in self._cmd(tmp_path, compile_activations=False)


def test_config_manager_defaults_false(tmp_path):
    cfg = ConfigManager(config_dir=str(tmp_path))
    assert cfg.get_default_training_config()["compile_activations"] is False
    assert cfg.get_default_generation_config()["compile_activations"] is False


def test_flask_load_forwards_compile_activations(tmp_path):
    from fluxflow_ui import app_flask

    with patch.object(
        app_flask.generation_worker, "load_model", return_value=(True, "ok")
    ) as mock_load:
        resp = app_flask.app.test_client().post(
            "/api/generation/load",
            json={"checkpoint_path": "x", "compile_activations": True},
        )
    assert resp.status_code == 200
    assert mock_load.call_args.kwargs["compile_activations"] is True
