"""Agent 注册表单元测试。"""

from aiops_agent.agents import ALL_AGENTS


class TestAgentRegistry:

    def test_required_agents_exist(self):
        names = {a["name"] for a in ALL_AGENTS}
        assert "planner" in names
        assert "worker" in names

    def test_planner_has_knowledge_tool(self):
        planner = next(a for a in ALL_AGENTS if a["name"] == "planner")
        assert "retrieve_knowledge" in planner["tools"]

    def test_worker_has_shell(self):
        worker = next(a for a in ALL_AGENTS if a["name"] == "worker")
        assert "shell" in worker["tools"]
