"""配置模块单元测试。"""

import os
import tempfile
from pathlib import Path

import pytest

from aiops_agent.config import Config


class TestConfig:

    def test_default_values(self):
        config = Config()
        assert config.llm_provider == "openai_compatible"
        assert config.api_key == ""
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4o-mini"
        assert config.max_tool_rounds == 10
        assert config.memory_strategy == "tiered"

    def test_validate(self):
        # 缺少 key
        errors = Config().validate()
        assert len(errors) > 0
        assert "OPENAI_API_KEY" in errors[0]

        # 无效 URL
        errors = Config(api_key="sk-valid-key-12345", base_url="not-a-url").validate()
        assert len(errors) > 0
        assert "BASE_URL" in errors[0]

        # 合法配置
        errors = Config(api_key="sk-valid-key-12345").validate()
        assert len(errors) == 0

    def test_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'OPENAI_API_KEY=sk-from-env\n'
                'OPENAI_MODEL=gpt-4\n'
                'MAX_TOOL_ROUNDS=5\n'
                'MEMORY_COMPACTION_ENABLED=false\n',
                encoding="utf-8",
            )
            # 用临时目录隔离，避免受项目根目录 .env 影响
            import os
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                config = Config.from_env(env_path)
                assert config.api_key == "sk-from-env"
                assert config.model == "gpt-4"
                assert config.max_tool_rounds == 5
                assert config.base_url == "https://api.openai.com/v1"  # 默认值
                assert config.memory_compaction_enabled is False
            finally:
                os.chdir(orig_cwd)
