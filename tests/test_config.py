"""配置模块单元测试。"""

import os
import tempfile
from pathlib import Path

import pytest

from aiops_agent.config import Config


class TestConfig:
    """Config 基础功能测试。"""

    def test_default_values(self):
        config = Config()
        assert config.llm_provider == "openai_compatible"
        assert config.api_key == ""
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4o-mini"
        assert config.max_tool_rounds == 10
        assert config.memory_strategy == "tiered"

    def test_validate_missing_api_key(self):
        config = Config()
        errors = config.validate()
        assert len(errors) > 0
        assert "OPENAI_API_KEY" in errors[0]

    def test_validate_invalid_url(self):
        config = Config(api_key="sk-valid-key-12345", base_url="not-a-url")
        errors = config.validate()
        assert len(errors) > 0
        assert "BASE_URL" in errors[0]

    def test_validate_success(self):
        config = Config(api_key="sk-valid-key-12345")
        errors = config.validate()
        assert len(errors) == 0

    def test_from_env(self):
        """测试从 .env 文件加载配置。"""
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'OPENAI_API_KEY=sk-from-env\n'
                'OPENAI_MODEL=gpt-4\n'
                'MAX_TOOL_ROUNDS=5\n',
                encoding="utf-8",
            )
            config = Config.from_env(env_path)
            assert config.api_key == "sk-from-env"
            assert config.model == "gpt-4"
            assert config.max_tool_rounds == 5
            # 未设置的字段应有默认值
            assert config.base_url == "https://api.openai.com/v1"

    def test_from_env_compaction_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'MEMORY_COMPACTION_ENABLED=false\n'
                'OPENAI_API_KEY=sk-test\n',
                encoding="utf-8",
            )
            config = Config.from_env(env_path)
            assert config.memory_compaction_enabled is False
