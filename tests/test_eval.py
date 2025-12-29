"""Tests for evaluation system."""

import pytest
import asyncio
from agentu import Agent, Tool, evaluate, EvalResult, FailedCase


@pytest.fixture
def test_agent():
    """Create agent with simple tools for testing."""
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    def echo(message: str) -> str:
        """Echo message back."""
        return f"Echo: {message}"
    
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"The weather in {city} is sunny and 72°F"
    
    agent = Agent(
        name="TestAgent",
        model="qwen3:latest",
        enable_memory=False  # Disable for faster tests
    ).with_tools([
        Tool(add),
        Tool(echo),
        Tool(get_weather)
    ])
    
    return agent


class TestEvaluate:
    """Test evaluate() function."""
    
    @pytest.mark.asyncio
    async def test_exact_match(self, test_agent):
        """Test exact matching."""
        cases = [
            {"ask": "Add 5 and 3", "expect": 8},
            {"ask": "Add 10 and 20", "expect": 30}
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.total == 2
        assert result.passed == 2
        assert result.failed == 0
        assert result.accuracy == 100.0
    
    @pytest.mark.asyncio
    async def test_substring_match(self, test_agent):
        """Test substring matching for strings."""
        cases = [
            {"ask": "Echo hello world", "expect": "hello"},
            {"ask": "Weather in SF", "expect": "sunny"}
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.passed == 2
        assert result.accuracy == 100.0
    
    @pytest.mark.asyncio
    async def test_failed_cases(self, test_agent):
        """Test that failures are tracked."""
        cases = [
            {"ask": "Add 5 and 3", "expect": 999},  # Wrong expectation
            {"ask": "Echo test", "expect": "wrong"}  # Won't match
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.passed == 0
        assert result.failed == 2
        assert len(result.failures) == 2
        assert result.accuracy == 0.0
    
    @pytest.mark.asyncio
    async def test_expect_tool(self, test_agent):
        """Test expected tool validation."""
        cases = [
            {"ask": "Add 5 and 3", "expect": 8, "expect_tool": "add"},
            {"ask": "Echo hello", "expect": "hello", "expect_tool": "echo"}
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.passed == 2
    
    @pytest.mark.asyncio
    async def test_wrong_tool_fails(self, test_agent):
        """Test that wrong tool usage fails."""
        cases = [
            {"ask": "Add 5 and 3", "expect": 8, "expect_tool": "wrong_tool"}
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.failed == 1
        assert "Wrong tool" in result.failures[0].reason
    
    @pytest.mark.asyncio
    async def test_custom_validator(self, test_agent):
        """Test custom validation function."""
        def custom_check(expected, actual):
            return actual > expected
        
        cases = [
            {
                "ask": "Add 5 and 3",
                "expect": 5,  # Actual will be 8, which is > 5
                "validator": custom_check
            }
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.passed == 1
    
    @pytest.mark.asyncio
    async def test_timeout(self, test_agent):
        """Test timeout handling."""
        cases = [
            {
                "ask": "Add 5 and 3",
                "expect": 8,
                "timeout": 0.001  # Very short timeout
            }
        ]
        
        result = await evaluate(test_agent, cases)
        
        # Should timeout or pass quickly
        assert result.total == 1
    
    @pytest.mark.asyncio
    async def test_max_turns(self, test_agent):
        """Test max turns validation."""
        cases = [
            {
                "ask": "Add 5 and 3",
                "expect": 8,
                "max_turns": 1  # Should pass in 1 turn
            }
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.passed == 1
        assert result.avg_turns <= 1
    
    @pytest.mark.asyncio
    async def test_avg_turns_calculation(self, test_agent):
        """Test average turns calculation."""
        cases = [
            {"ask": "Add 5 and 3", "expect": 8},
            {"ask": "Echo test", "expect": "test"},
            {"ask": "Weather in NYC", "expect": "sunny"}
        ]
        
        result = await evaluate(test_agent, cases)
        
        assert result.avg_turns > 0
        assert result.avg_turns <= 3  # Should be reasonable


class TestEvalResult:
    """Test EvalResult dataclass."""
    
    def test_accuracy_calculation(self):
        """Test accuracy percentage calculation."""
        result = EvalResult(passed=8, failed=2, total=10)
        assert result.accuracy == 80.0
        
        result2 = EvalResult(passed=10, failed=0, total=10)
        assert result2.accuracy == 100.0
        
        result3 = EvalResult(passed=0, failed=10, total=10)
        assert result3.accuracy == 0.0
    
    def test_accuracy_with_zero_total(self):
        """Test accuracy with zero test cases."""
        result = EvalResult(passed=0, failed=0, total=0)
        assert result.accuracy == 0.0
    
    def test_str_representation(self):
        """Test string representation."""
        result = EvalResult(
            passed=8,
            failed=2,
            total=10,
            avg_turns=1.5,
            duration=2.5
        )
        
        str_repr = str(result)
        assert "8/10" in str_repr
        assert "80.0%" in str_repr
        assert "1.5" in str_repr
    
    def test_str_with_failures(self):
        """Test string representation with failures."""
        failure = FailedCase(
            query="Test query",
            expected="expected",
            actual="actual",
            reason="Mismatch"
        )
        
        result = EvalResult(
            passed=0,
            failed=1,
            total=1,
            failures=[failure]
        )
        
        str_repr = str(result)
        assert "Failures:" in str_repr
        assert "Test query" in str_repr
        assert "expected" in str_repr
        assert "actual" in str_repr


class TestFailedCase:
    """Test FailedCase dataclass."""
    
    def test_creation(self):
        """Test creating a failed case."""
        failure = FailedCase(
            query="Add 5 and 3",
            expected=8,
            actual=10,
            reason="Wrong result",
            tool_used="add",
            turns=2
        )
        
        assert failure.query == "Add 5 and 3"
        assert failure.expected == 8
        assert failure.actual == 10
        assert failure.reason == "Wrong result"
        assert failure.tool_used == "add"
        assert failure.turns == 2
    
    def test_optional_fields(self):
        """Test optional fields have defaults."""
        failure = FailedCase(
            query="Test",
            expected="expected",
            actual="actual",
            reason="test"
        )
        
        assert failure.tool_used is None
        assert failure.turns == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
