import pytest
from pydantic import BaseModel
from agentu._core.result import InferResult


class CitySummary(BaseModel):
    name: str
    population: int


class TestInferResult:
    def test_dict_compatibility(self):
        r = InferResult(text="London is a city", structured=CitySummary(name="London", population=9000000), turns=2)
        assert r["text_response"] == "London is a city"
        assert r["structured"].name == "London"
        assert r["turns"] == 2
        assert "text_response" in r
        assert r.get("missing", 42) == 42

    def test_typed_attribute_access(self):
        summary = CitySummary(name="Tokyo", population=14000000)
        r: InferResult[CitySummary] = InferResult(text="Tokyo info", structured=summary)
        assert r.data.name == "Tokyo"
        assert r.structured.population == 14000000
        assert str(r) == "Tokyo info"

    def test_mutation_dict_compatibility(self):
        r = InferResult(text="Initial")
        r["text_response"] = "Updated"
        assert r.text == "Updated"
        assert r["text_response"] == "Updated"
        assert isinstance(r, dict)

    @pytest.mark.asyncio
    async def test_agent_infer_returns_infer_result_for_output_type_and_schema(self):
        from unittest.mock import AsyncMock
        from agentu import Agent

        agent = Agent("test-agent", enable_memory=False, auto_discover_rules=False)
        agent._call_llm = AsyncMock(return_value='{"name": "Kyoto", "population": 1460000}')

        res_type: InferResult[CitySummary] = await agent.infer("City info", output_type=CitySummary)
        assert isinstance(res_type, InferResult)
        assert isinstance(res_type, dict)
        assert res_type.data.name == "Kyoto"
        assert res_type.structured.population == 1460000
        assert res_type["structured"].name == "Kyoto"
        assert res_type["attempts"] == 1

        res_schema: InferResult[CitySummary] = await agent.infer("City info", output_schema=CitySummary)
        assert isinstance(res_schema, InferResult)
        assert res_schema.data.name == "Kyoto"
        assert res_schema["structured"].population == 1460000

