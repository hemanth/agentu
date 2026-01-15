"""Tests for workflow checkpoint and resume functionality."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agentu.workflow import (
    Step, SequentialStep, ParallelStep,
    WorkflowCheckpoint, _save_checkpoint, _load_checkpoint,
    resume_workflow
)


class TestWorkflowCheckpoint:
    """Test WorkflowCheckpoint dataclass."""
    
    def test_checkpoint_creation(self):
        cp = WorkflowCheckpoint(
            workflow_id="abc123",
            created_at=1000.0,
            updated_at=1000.0,
            current_step=0,
            total_steps=3
        )
        assert cp.workflow_id == "abc123"
        assert cp.status == "in_progress"
        assert cp.error is None
    
    def test_to_dict(self):
        cp = WorkflowCheckpoint(
            workflow_id="test",
            created_at=1000.0,
            updated_at=1001.0,
            current_step=1,
            total_steps=3,
            completed_steps=[{"step_index": 0}]
        )
        d = cp.to_dict()
        assert d["workflow_id"] == "test"
        assert d["current_step"] == 1
        assert len(d["completed_steps"]) == 1
    
    def test_from_dict(self):
        data = {
            "workflow_id": "xyz",
            "created_at": 1000.0,
            "updated_at": 1001.0,
            "current_step": 2,
            "total_steps": 5,
            "completed_steps": [],
            "status": "in_progress",
            "error": None
        }
        cp = WorkflowCheckpoint.from_dict(data)
        assert cp.workflow_id == "xyz"
        assert cp.current_step == 2


class TestCheckpointIO:
    """Test checkpoint save/load functions."""
    
    def test_save_and_load(self, tmp_path):
        cp = WorkflowCheckpoint(
            workflow_id="save_test",
            created_at=1000.0,
            updated_at=1001.0,
            current_step=1,
            total_steps=3,
            completed_steps=[{"result": "test"}]
        )
        
        filepath = _save_checkpoint(cp, str(tmp_path))
        assert Path(filepath).exists()
        
        loaded = _load_checkpoint(filepath)
        assert loaded.workflow_id == "save_test"
        assert loaded.current_step == 1
        assert len(loaded.completed_steps) == 1


class TestResumeWorkflow:
    """Test resume_workflow function."""
    
    @pytest.mark.asyncio
    async def test_resume_completed_workflow(self, tmp_path):
        """Test resuming a completed workflow returns last result."""
        cp = WorkflowCheckpoint(
            workflow_id="done",
            created_at=1000.0,
            updated_at=1001.0,
            current_step=3,
            total_steps=3,
            completed_steps=[
                {"result": "step1"},
                {"result": "step2"},
                {"result": "final"}
            ],
            status="completed"
        )
        
        filepath = _save_checkpoint(cp, str(tmp_path))
        result = await resume_workflow(filepath)
        
        assert result == "final"
    
    @pytest.mark.asyncio
    async def test_resume_in_progress_workflow(self, tmp_path):
        """Test resuming an in-progress workflow returns checkpoint info."""
        cp = WorkflowCheckpoint(
            workflow_id="partial",
            created_at=1000.0,
            updated_at=1001.0,
            current_step=1,
            total_steps=3,
            completed_steps=[{"result": "step1"}],
            status="in_progress"
        )
        
        filepath = _save_checkpoint(cp, str(tmp_path))
        result = await resume_workflow(filepath)
        
        assert result["resumed_from"] == 1
        assert result["context"] == "step1"
        assert len(result["completed_steps"]) == 1


class TestSequentialStepWithCheckpoint:
    """Test SequentialStep with checkpointing."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.infer = AsyncMock(side_effect=["result1", "result2", "result3"])
        return agent
    
    def test_flatten_steps(self, mock_agent):
        """Test that nested sequential steps are flattened."""
        step1 = Step(mock_agent, "task1")
        step2 = Step(mock_agent, "task2")
        step3 = Step(mock_agent, "task3")
        
        workflow = step1 >> step2 >> step3
        
        # Get flattened steps
        flattened = workflow._flatten_steps()
        assert len(flattened) == 3
    
    @pytest.mark.asyncio
    async def test_sequential_with_checkpoint(self, mock_agent, tmp_path):
        """Test that checkpoints are saved during sequential execution."""
        step1 = Step(mock_agent, "task1")
        step2 = Step(mock_agent, "task2")
        
        workflow = step1 >> step2
        
        checkpoint_dir = str(tmp_path / "checkpoints")
        result = await workflow.run(checkpoint=checkpoint_dir)
        
        # Result should include checkpoint_path
        assert "checkpoint_path" in result
        assert "result" in result
        assert Path(result["checkpoint_path"]).exists()
        
        # Load and verify
        with open(result["checkpoint_path"]) as f:
            data = json.load(f)
        
        assert data["status"] == "completed"
        assert data["current_step"] == 2
        assert len(data["completed_steps"]) == 2
    
    @pytest.mark.asyncio
    async def test_sequential_with_custom_workflow_id(self, mock_agent, tmp_path):
        """Test that custom workflow_id is used in checkpoint filename."""
        step1 = Step(mock_agent, "task1")
        step2 = Step(mock_agent, "task2")
        
        workflow = step1 >> step2
        
        checkpoint_dir = str(tmp_path / "checkpoints")
        result = await workflow.run(checkpoint=checkpoint_dir, workflow_id="my-custom-id")
        
        # Verify custom ID is in the path
        assert "my-custom-id" in result["checkpoint_path"]
        assert Path(result["checkpoint_path"]).exists()


class TestStepWithCheckpoint:
    """Test Step class with checkpoint parameter."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.infer = AsyncMock(return_value={"result": "success"})
        return agent
    
    @pytest.mark.asyncio
    async def test_single_step_ignores_checkpoint(self, mock_agent):
        """Test that single step ignores checkpoint parameter (no-op)."""
        step = Step(mock_agent, "task")
        result = await step.run(checkpoint="./checkpoints")
        
        assert result == {"result": "success"}


class TestParallelStepWithCheckpoint:
    """Test ParallelStep with checkpoint parameter."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.infer = AsyncMock(return_value={"result": "parallel"})
        return agent
    
    @pytest.mark.asyncio
    async def test_parallel_step_with_checkpoint(self, mock_agent):
        """Test parallel step execution (checkpoint is currently no-op for parallel)."""
        step1 = Step(mock_agent, "task1")
        step2 = Step(mock_agent, "task2")
        
        workflow = step1 & step2
        results = await workflow.run(checkpoint="./checkpoints")
        
        assert len(results) == 2
