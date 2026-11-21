"""_load_v100_fallback builds the v0.10.0 VAE at the checkpoint's own widths.

Wide v0.10.0 checkpoints (non-default encoder/decoder/SPADE widths) previously
failed with IncompatibleCheckpointError because the fallback built the VAE at
module-default widths.
"""

from unittest.mock import MagicMock

import safetensors.torch
import torch

from fluxflow_ui.utils.generation_worker import GenerationWorker


def test_v100_fallback_loads_wide_vae(tmp_path, monkeypatch):
    import fluxflow.models.v100.flow as v100_flow
    import fluxflow_ui.utils.generation_worker as gw
    from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

    enc, dec, hid = [24, 40], [40, 24], [64, 48]
    src_comp = FluxCompressor_v100(d_model=16, downscales=2, max_hw=64, encoder_channels=enc)
    src_exp = FluxExpander_v100(
        d_model=16, upscales=2, max_hw=64, decoder_channels=dec, spade_hidden=hid
    )
    state = {f"diffuser.compressor.{k}": v for k, v in src_comp.state_dict().items()}
    state.update({f"diffuser.expander.{k}": v for k, v in src_exp.state_dict().items()})
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(state, str(path), metadata={"activation_type": "bezier"})

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
    ok, _ = worker._load_v100_fallback(
        str(path), vae_dim=16, feature_maps_dim=32, text_embedding_dim=32
    )

    assert ok
    assert worker.diffuser.compressor.stage_channels[1:] == enc
    loaded = worker.diffuser.state_dict()
    for key, value in state.items():
        assert torch.equal(loaded[key.removeprefix("diffuser.")], value), key
