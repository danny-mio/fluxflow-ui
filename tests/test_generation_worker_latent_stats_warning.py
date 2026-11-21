"""Final review M4: GenerationWorker warns after load on uncalibrated/mismatched latent stats."""

import logging
from unittest.mock import MagicMock, patch

import safetensors.torch
import torch

from fluxflow_ui.utils.generation_worker import GenerationWorker

GW = "fluxflow_ui.utils.generation_worker"


def _v100_ckpt(tmp_path, calibrate: bool) -> str:
    from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

    comp = FluxCompressor_v100(d_model=16, downscales=2, max_hw=64)
    exp = FluxExpander_v100(d_model=16, upscales=2, max_hw=64)
    if calibrate:
        for m in (comp, exp):
            m.set_latent_stats(torch.full((32,), 0.1), torch.full((32,), 2.0))
    state = {f"diffuser.compressor.{k}": v for k, v in comp.state_dict().items()}
    state.update({f"diffuser.expander.{k}": v for k, v in exp.state_dict().items()})
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(state, str(path), metadata={"activation_type": "bezier"})
    return str(path)


def _load_v100(path, monkeypatch) -> GenerationWorker:
    import fluxflow.models.v100.flow as v100_flow

    import fluxflow_ui.utils.generation_worker as gw

    monkeypatch.setattr(v100_flow, "FluxFlowProcessor_v100", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        gw.FluxPipeline,
        "_detect_config",
        staticmethod(lambda state_dict: {"vae_dim": 16, "downscales": 2, "flow_dim": 32}),
    )
    monkeypatch.setattr(
        gw,
        "AutoTokenizer",
        MagicMock(from_pretrained=MagicMock(return_value=MagicMock(pad_token="x"))),
    )
    monkeypatch.setattr(gw, "BertTextEncoder", MagicMock())
    worker = object.__new__(GenerationWorker)
    worker.device = "cpu"
    ok, _ = worker._load_v100_fallback(path, vae_dim=16, feature_maps_dim=32, text_embedding_dim=32)
    assert ok
    return worker


def test_v100_fallback_warns_when_uncalibrated(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=GW):
        _load_v100(_v100_ckpt(tmp_path, calibrate=False), monkeypatch)
    assert "uncalibrated" in caplog.text


def test_v100_fallback_silent_when_calibrated(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=GW):
        worker = _load_v100(_v100_ckpt(tmp_path, calibrate=True), monkeypatch)
    assert worker.diffuser.latent_stats_calibrated
    assert "uncalibrated" not in caplog.text and "differ" not in caplog.text


def test_versioned_paths_check_after_load(tmp_path):
    for path in (tmp_path, _v100_ckpt(tmp_path, calibrate=False)):
        pipeline = MagicMock()
        pipeline.to.return_value = pipeline
        worker = object.__new__(GenerationWorker)
        worker.device = torch.device("cpu")
        with (
            patch(f"{GW}.load_versioned_checkpoint", return_value=pipeline),
            patch(f"{GW}.AutoTokenizer", MagicMock()),
            patch(f"{GW}.BertTextEncoder", MagicMock()),
            patch.object(GenerationWorker, "_detect_display_version", return_value="0.10.0"),
            patch(f"{GW}.warn_if_latent_stats_uncalibrated", create=True) as warn,
        ):
            ok, msg = worker.load_model(str(path))
        assert ok, msg
        warn.assert_called_once()
        assert warn.call_args.args[0] is pipeline
