"""Tests for CLI checkpoint and resume helpers."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from cane_personality.cli import _checkpoint_path, _load_checkpoint, _append_checkpoint


class TestCheckpoint:
    def test_checkpoint_path_from_output_json(self):
        """Test that _checkpoint_path derives the right path from output_json."""
        args = Namespace(output_json="results/profile.json")
        result = _checkpoint_path(args)
        assert result == Path("results/profile.checkpoint.jsonl")

    def test_checkpoint_path_default(self):
        """When output_json is None, use the default checkpoint path."""
        args = Namespace(output_json=None)
        result = _checkpoint_path(args)
        assert result == Path(".cane_checkpoint.jsonl")

    def test_load_checkpoint_empty(self, tmp_path):
        """Loading a non-existent checkpoint returns empty dict."""
        ckpt = tmp_path / "missing.checkpoint.jsonl"
        result = _load_checkpoint(ckpt)
        assert result == {}

    def test_append_and_load_checkpoint(self, tmp_path):
        """Append a result, load it back, verify it matches."""
        ckpt = tmp_path / "test.checkpoint.jsonl"
        entry = {
            "question": "What is 2+2?",
            "expected_answer": "4",
            "agent_answer": "4",
            "score": 95,
            "status": "pass",
        }
        _append_checkpoint(ckpt, entry)
        loaded = _load_checkpoint(ckpt)
        assert "What is 2+2?" in loaded
        assert loaded["What is 2+2?"]["score"] == 95
        assert loaded["What is 2+2?"]["status"] == "pass"

    def test_checkpoint_cleanup(self, tmp_path):
        """After appending, verify the file exists, then delete it."""
        ckpt = tmp_path / "cleanup.checkpoint.jsonl"
        entry = {
            "question": "Test question",
            "expected_answer": "answer",
            "agent_answer": "answer",
            "score": 80,
            "status": "pass",
        }
        _append_checkpoint(ckpt, entry)
        assert ckpt.exists()
        ckpt.unlink()
        assert not ckpt.exists()
