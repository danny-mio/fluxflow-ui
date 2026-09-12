"""Tests for GenerationWorker activation_type (Bezier/Pade) detection.

_load_v100_fallback previously always constructed FluxCompressor_v100/
FluxFlowProcessor_v100/FluxExpander_v100 with the implicit "bezier" default,
silently ignoring whichever activation family a checkpoint was actually
trained with. These tests cover the fix: reading activation_type back from
the checkpoint's safetensors metadata and threading it through.
"""

from unittest.mock import MagicMock

import safetensors.torch
import torch

from fluxflow_ui.utils.generation_worker import GenerationWorker


def _make_worker():
    return object.__new__(GenerationWorker)


class TestDetectActivationType:
    def test_defaults_to_bezier_when_no_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"w": torch.zeros(2)}, str(path))

        worker = _make_worker()
        assert worker._detect_activation_type(str(path)) == "bezier"

    def test_reads_pade_from_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file(
            {"w": torch.zeros(2)}, str(path), metadata={"activation_type": "pade"}
        )

        worker = _make_worker()
        assert worker._detect_activation_type(str(path)) == "pade"

    def test_resolves_directory_checkpoint_to_flxflow_final(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        path = ckpt_dir / "flxflow_final.safetensors"
        safetensors.torch.save_file(
            {"w": torch.zeros(2)}, str(path), metadata={"activation_type": "pade"}
        )

        worker = _make_worker()
        assert worker._detect_activation_type(str(ckpt_dir)) == "pade"

    def test_missing_file_defaults_to_bezier(self, tmp_path):
        worker = _make_worker()
        assert worker._detect_activation_type(str(tmp_path / "does_not_exist.safetensors")) == (
            "bezier"
        )


class TestLoadV100FallbackThreadsActivationType:
    def test_v100_fallback_passes_detected_activation_type(self, tmp_path, monkeypatch):
        import fluxflow_ui.utils.generation_worker as gw

        path = tmp_path / "model.safetensors"
        diffuser_state = {
            "diffuser.compressor.latent_proj.0.0.weight": torch.zeros(4, 4, 1, 1),
        }
        safetensors.torch.save_file(diffuser_state, str(path), metadata={"activation_type": "pade"})

        captured = {}

        class _FakeCompressor:
            def __init__(self, **kwargs):
                captured["compressor"] = kwargs

        class _FakeFlow:
            def __init__(self, **kwargs):
                captured["flow"] = kwargs

        class _FakeExpander:
            def __init__(self, **kwargs):
                captured["expander"] = kwargs

        # _load_v100_fallback imports these locally (`from fluxflow.models.v100.vae
        # import ...`) at call time, so patching gw.FluxCompressor_v100 etc. has no
        # effect -- must patch the source modules themselves.
        import fluxflow.models.v100.flow as v100_flow
        import fluxflow.models.v100.vae as v100_vae

        monkeypatch.setattr(v100_vae, "FluxCompressor_v100", _FakeCompressor)
        monkeypatch.setattr(v100_flow, "FluxFlowProcessor_v100", _FakeFlow)
        monkeypatch.setattr(v100_vae, "FluxExpander_v100", _FakeExpander)
        monkeypatch.setattr(
            gw.FluxPipeline,
            "_detect_config",
            staticmethod(
                lambda state_dict: {"vae_dim": 4, "downscales": 2, "flow_dim": 8}
            ),
        )
        monkeypatch.setattr(
            gw,
            "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=MagicMock(pad_token="x"))),
        )
        monkeypatch.setattr(gw, "BertTextEncoder", MagicMock())

        worker = _make_worker()
        worker.device = "cpu"

        try:
            worker._load_v100_fallback(str(path), vae_dim=4, feature_maps_dim=8, text_embedding_dim=8)
        except Exception:
            pass  # FluxPipeline(...) construction with fakes may not fully succeed; not the point of this test

        assert captured.get("compressor", {}).get("activation_type") == "pade"
        assert captured.get("flow", {}).get("activation_type") == "pade"
        assert captured.get("expander", {}).get("activation_type") == "pade"
