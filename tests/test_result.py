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
