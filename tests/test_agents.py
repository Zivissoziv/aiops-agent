"""Agent 注册表单元测试。"""

from aiops_agent.agents import ALL_AGENTS


class TestAgentRegistry:
    """Agent 注册表测试。"""

    def test_all_agents_not_empty(self):
        assert len(ALL_AGENTS) > 0

    def test_planner_exists(self):
        names = [a["name"] for a in ALL_AGENTS]
        assert "planner" in names

    def test_worker_exists(self):
        names = [a["name"] for a in ALL_AGENTS]
        assert "worker" in names

    def test_each_agent_has_required_fields(self):
        for agent in ALL_AGENTS:
            assert "name" in agent
            assert "tools" in agent
            assert isinstance(agent["tools"], list)
    def test_planner_has_knowledge_tool(self):
        """planner 现在有 retrieve_knowledge 工具（用于查知识库后直接回答）。"""
        planner = [a for a in ALL_AGENTS if a["name"] == "planner"][0]
        assert "retrieve_knowledge" in planner["tools"]

    def test_worker_has_tools(self):
        worker = next(a for a in ALL_AGENTS if a["name"] == "worker")
        assert len(worker["tools"]) > 0
        assert "shell" in worker["tools"]
