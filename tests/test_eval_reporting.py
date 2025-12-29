"""Additional tests for reporting features."""

import pytest
import json
from agentu import EvalResult, FailedCase


class TestReporting:
    """Test reporting features."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = EvalResult(
            passed=8,
            failed=2,
            total=10,
            avg_turns=1.5,
            duration=2.5
        )
        
        data = result.to_dict()
        
        assert data["passed"] == 8
        assert data["failed"] == 2
        assert data["total"] == 10
        assert data["accuracy"] == 80.0
        assert data["avg_turns"] == 1.5
        assert data["duration"] == 2.5
    
    def test_to_dict_with_failures(self):
        """Test dictionary conversion with failures."""
        failure = FailedCase(
            query="Test",
            expected="expected",
            actual="actual",
            reason="Mismatch",
            tool_used="test_tool",
            turns=2
        )
        
        result = EvalResult(
            passed=0,
            failed=1,
            total=1,
            failures=[failure]
        )
        
        data = result.to_dict()
        
        assert len(data["failures"]) == 1
        assert data["failures"][0]["query"] == "Test"
        assert data["failures"][0]["tool_used"] == "test_tool"
        assert data["failures"][0]["turns"] == 2
    
    def test_to_json(self):
        """Test JSON export."""
        result = EvalResult(
            passed=5,
            failed=0,
            total=5
        )
        
        json_str = result.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["passed"] == 5
        assert data["accuracy"] == 100.0
    
    def test_str_with_colors(self):
        """Test string output includes color codes."""
        result = EvalResult(
            passed=8,
            failed=2,
            total=10,
            avg_turns=1.5
        )
        
        output = str(result)
        
        # Check for basic content
        assert "8/10" in output
        assert "Results:" in output
        assert "Passed:" in output
    
    def test_str_shows_failures(self):
        """Test that failures are displayed."""
        failure = FailedCase(
            query="Add 5 and 3",
            expected=8,
            actual=10,
            reason="Wrong calculation",
            tool_used="add",
            turns=1
        )
        
        result = EvalResult(
            passed=0,
            failed=1,
            total=1,
            failures=[failure]
        )
        
        output = str(result)
        
        assert "Failures:" in output
        assert "Add 5 and 3" in output
        assert "Wrong calculation" in output
        assert "add" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
