"""Tests for TrainingRunner."""

import queue
import threading
from unittest.mock import MagicMock, patch

from fluxflow_ui.utils.training_runner import TrainingRunner


class TestTrainingRunnerInit:
    """Tests for TrainingRunner initialization."""

    def test_initializes_with_defaults(self):
        """Should initialize with correct default values."""
        runner = TrainingRunner()

        assert runner.process is None
        assert runner.is_running is False
        assert runner.thread is None
        assert runner.temp_config_file is None
        assert isinstance(runner.output_queue, queue.Queue)

    def test_is_running_is_thread_safe(self):
        """is_running property should be thread-safe."""
        runner = TrainingRunner()

        # Should be accessible without errors
        assert runner.is_running is False


class TestStartTraining:
    """Tests for starting training."""

    def test_returns_false_if_already_running(self):
        """Should return False if training is already running."""
        runner = TrainingRunner()
        runner._is_running = True

        result = runner.start_training({"data_path": "/test"})

        assert result is False

    @patch("subprocess.Popen")
    def test_starts_process_with_correct_command(self, mock_popen):
        """Should start subprocess with fluxflow-train command."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        config = {
            "data_path": "/path/to/data",
            "batch_size": 4,
            "lr": 1e-4,
        }

        result = runner.start_training(config)

        assert result is True
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "fluxflow-train"
        assert "--data_path" in cmd
        assert "/path/to/data" in cmd

    @patch("subprocess.Popen")
    def test_sets_is_running_to_true(self, mock_popen):
        """Should set is_running to True after starting."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        runner.start_training({})

        assert runner.is_running is True

    @patch("subprocess.Popen")
    def test_includes_all_config_options(self, mock_popen):
        """Should include all config options in command."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        config = {
            "vae_dim": 128,
            "batch_size": 8,
            "use_fp16": True,
            "train_vae": True,
        }

        runner.start_training(config)

        cmd = mock_popen.call_args[0][0]
        assert "--vae_dim" in cmd
        assert "128" in cmd
        assert "--batch_size" in cmd
        assert "8" in cmd
        assert "--use_fp16" in cmd
        assert "--train_vae" in cmd

    @patch("subprocess.Popen")
    def test_returns_false_on_file_not_found(self, mock_popen):
        """Should return False if command not found."""
        mock_popen.side_effect = FileNotFoundError("Command not found")
        runner = TrainingRunner()

        result = runner.start_training({})

        # Should return False due to FileNotFoundError
        assert result is False

    @patch("subprocess.Popen")
    def test_creates_temp_config_for_optimizers(self, mock_popen, tmp_path):
        """Should create temp config file for optimizers/schedulers."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        config = {
            "output_path": str(tmp_path),
            "optimizers": {"flow": {"type": "AdamW"}},
            "schedulers": {"flow": {"type": "CosineAnnealingLR"}},
        }

        runner.start_training(config)

        assert runner.temp_config_file is not None
        assert runner.temp_config_file.endswith("optim_sched_config.json")


class TestStopTraining:
    """Tests for stopping training."""

    def test_returns_false_if_not_running(self):
        """Should return False if not running."""
        runner = TrainingRunner()

        result = runner.stop_training()

        assert result is False

    @patch("subprocess.Popen")
    def test_terminates_process(self, mock_popen):
        """Should terminate the subprocess."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        runner.start_training({})

        result = runner.stop_training()

        assert result is True
        mock_process.terminate.assert_called_once()

    @patch("subprocess.Popen")
    def test_sets_is_running_to_false(self, mock_popen):
        """Should set is_running to False after stopping."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        runner.start_training({})
        runner.stop_training()

        assert runner.is_running is False

    @patch("subprocess.Popen")
    def test_kills_process_on_timeout(self, mock_popen):
        """Should kill process if terminate times out."""
        import subprocess

        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        runner.start_training({})

        result = runner.stop_training()

        assert result is True
        mock_process.kill.assert_called_once()


class TestGetOutput:
    """Tests for getting training output."""

    def test_returns_empty_list_when_no_output(self):
        """Should return empty list when queue is empty."""
        runner = TrainingRunner()

        output = runner.get_output()

        assert output == []

    def test_returns_all_queued_lines(self):
        """Should return all lines from queue."""
        runner = TrainingRunner()
        runner.output_queue.put("line 1")
        runner.output_queue.put("line 2")
        runner.output_queue.put("line 3")

        output = runner.get_output()

        assert output == ["line 1", "line 2", "line 3"]

    def test_empties_queue_after_get(self):
        """Should empty queue after getting output."""
        runner = TrainingRunner()
        runner.output_queue.put("test")

        runner.get_output()
        output = runner.get_output()

        assert output == []


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_removes_temp_file(self, tmp_path):
        """Should remove temp config file on cleanup."""
        runner = TrainingRunner()

        # Create a temp file
        temp_file = tmp_path / "temp_config.json"
        temp_file.write_text("{}")
        runner.temp_config_file = str(temp_file)

        runner._cleanup()

        assert not temp_file.exists()

    def test_cleanup_handles_missing_file(self):
        """Should handle missing temp file gracefully."""
        runner = TrainingRunner()
        runner.temp_config_file = "/nonexistent/file.json"

        # Should not raise
        runner._cleanup()

    def test_cleanup_handles_none_temp_file(self):
        """Should handle None temp file gracefully."""
        runner = TrainingRunner()
        runner.temp_config_file = None

        # Should not raise
        runner._cleanup()


class TestThreadSafety:
    """Tests for thread safety."""

    @patch("subprocess.Popen")
    def test_concurrent_start_calls(self, mock_popen):
        """Should handle concurrent start calls safely."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_popen.return_value = mock_process

        runner = TrainingRunner()
        results = []
        lock = threading.Lock()

        def try_start():
            result = runner.start_training({})
            with lock:
                results.append(result)

        threads = [threading.Thread(target=try_start) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed, most should fail
        # Due to thread timing, exact counts can vary
        assert results.count(True) >= 1
        assert len(results) == 5

    def test_is_running_consistent_across_threads(self):
        """is_running property should be thread-safe to read."""
        runner = TrainingRunner()

        # Manually set internal state to test read consistency
        with runner._lock:
            runner._is_running = True

        values = []
        lock = threading.Lock()

        def check_running():
            for _ in range(100):
                val = runner.is_running
                with lock:
                    values.append(val)

        threads = [threading.Thread(target=check_running) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All values should be True since we set _is_running manually
        assert all(v is True for v in values)
