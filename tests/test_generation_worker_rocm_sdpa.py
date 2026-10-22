"""GenerationWorker disables broken ROCm SDPA backends for its device at init."""

import logging
from unittest.mock import patch

from fluxflow_ui.utils.generation_worker import GenerationWorker

PROBE = "fluxflow_ui.utils.generation_worker.disable_broken_rocm_sdpa"


def test_probe_called_with_worker_device():
    with patch(PROBE, return_value=[]) as probe:
        worker = GenerationWorker()
    probe.assert_called_once_with(worker.device)


def test_disabled_backends_logged(caplog):
    with patch(PROBE, return_value=["flash"]), caplog.at_level(logging.INFO):
        GenerationWorker()
    assert "flash" in caplog.text


def test_no_log_when_nothing_disabled(caplog):
    with patch(PROBE, return_value=[]), caplog.at_level(logging.INFO):
        GenerationWorker()
    assert "SDPA" not in caplog.text
