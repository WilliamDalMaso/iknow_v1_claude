from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.config import load_project_env, model_name


def test_load_project_env_sets_missing_values() -> None:
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    old_fast = os.environ.pop("OPENAI_MODEL_FAST", None)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\nOPENAI_MODEL_FAST=test-fast\n", encoding="utf-8")

            load_project_env(env_path)

            assert os.environ["OPENAI_API_KEY"] == "test-key"
            assert os.environ["OPENAI_MODEL_FAST"] == "test-fast"
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
        if old_fast is not None:
            os.environ["OPENAI_MODEL_FAST"] = old_fast
        else:
            os.environ.pop("OPENAI_MODEL_FAST", None)


def test_model_name_requires_explicit_value() -> None:
    old_fast = os.environ.pop("OPENAI_MODEL_FAST", None)
    try:
        try:
            model_name("fast")
        except RuntimeError as error:
            assert "OPENAI_MODEL_FAST is not set" in str(error)
        else:
            raise AssertionError("model_name should require explicit model configuration")
    finally:
        if old_fast is not None:
            os.environ["OPENAI_MODEL_FAST"] = old_fast


def test_model_name_reads_explicit_value() -> None:
    old_fast = os.environ.get("OPENAI_MODEL_FAST")
    try:
        os.environ["OPENAI_MODEL_FAST"] = "test-model"
        assert model_name("fast") == "test-model"
    finally:
        if old_fast is None:
            os.environ.pop("OPENAI_MODEL_FAST", None)
        else:
            os.environ["OPENAI_MODEL_FAST"] = old_fast
