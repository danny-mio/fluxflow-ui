"""Tests for the shared DEFAULT_MAX_TEXT_LENGTH default in GenerationWorker (v0.10.0).

Generation previously hardcoded max_length=512 while training tokenized
captions to 32 -- a silent mismatch. generation_worker.py now imports the
same constant training reads (via fluxflow.text_length), so the two call
sites in GenerationWorker.generate_image can't diverge from it again.
"""

from unittest.mock import MagicMock, patch

import pytest

from fluxflow_ui.utils.generation_worker import DEFAULT_MAX_TEXT_LENGTH, GenerationWorker


def _make_worker(device_type: str = "mps"):
    worker = object.__new__(GenerationWorker)
    worker.diffuser = MagicMock()
    worker.text_encoder = MagicMock()
    worker.tokenizer = MagicMock(side_effect=RuntimeError("stop-here"))
    worker.device = MagicMock()
    worker.device.type = device_type
    worker.config = {}
    return worker


def test_default_is_32():
    assert DEFAULT_MAX_TEXT_LENGTH == 32


class TestGenerateImageUsesSharedDefault:
    def test_tokenizer_called_with_shared_default(self):
        worker = _make_worker()
        with patch("torch.mps.empty_cache"):
            image, message = worker.generate_image("a prompt")
        assert image is None
        _, kwargs = worker.tokenizer.call_args
        assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH


class TestGenerateImageDeviceScoping:
    """Regression test for a conditional `import torch` inside
    `if self.device.type == "mps":` in generate_image. Python makes any
    name imported/assigned anywhere in a function local to that whole
    function, so on non-MPS devices (CPU/CUDA/ROCm) the module-level
    `torch` was shadowed and code earlier in the function that referenced
    `torch` raised UnboundLocalError before ever reaching the tokenizer.
    """

    def test_generate_image_does_not_unbound_local_error_on_cpu(self):
        worker = _make_worker(device_type="cpu")
        image, message = worker.generate_image("a prompt")
        assert image is None
        assert "UnboundLocalError" not in message
        assert "cannot access local variable" not in message
        # Must reach the (mocked) tokenizer call, not fail earlier on `torch`.
        assert message == "Generation failed: stop-here"
