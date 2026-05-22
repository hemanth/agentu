"""Tests for codemode feature.

Tests the code mode execution path where the LLM writes Python code
to call tools instead of making individual JSON tool calls.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agentu import Agent, Tool


# ─── Agent Configuration Tests ───


def test_codemode_default_false():
    """Test that codemode defaults to False."""
    agent = Agent(name="test_agent")
    assert agent.codemode is False


def test_codemode_init_flag():
    """Test setting codemode via Agent constructor."""
    agent = Agent(name="test_agent", codemode=True)
    assert agent.codemode is True


def test_codemode_via_sandbox():
    """Test enabling codemode via with_sandbox."""
    agent = Agent(name="test_agent")
    agent.with_sandbox(codemode=True)
    assert agent.codemode is True


def test_codemode_sandbox_without_flag():
    """Test that with_sandbox doesn't enable codemode by default."""
    agent = Agent(name="test_agent")
    agent.with_sandbox()
    assert agent.codemode is False


# ─── Type Stub Generation Tests ───


def test_generate_type_stubs_empty():
    """Test type stub generation with no tools."""
    agent = Agent(name="test_agent")
    stubs = agent._generate_type_stubs()
    assert "class tools:" in stubs
    assert "pass" in stubs


def test_generate_type_stubs_single_tool():
    """Test type stub generation with a single tool."""
    def search(query: str) -> str:
        """Search for information."""
        return query

    agent = Agent(name="test_agent")
    agent.with_tools([search])
    stubs = agent._generate_type_stubs()

    assert "class tools:" in stubs
    assert "def search" in stubs
    assert "query" in stubs
    assert "Search for information" in stubs


def test_generate_type_stubs_multiple_tools():
    """Test type stub generation with multiple tools."""
    def search(query: str) -> str:
        """Search the web."""
        return query

    def calculator(x: int, y: int, operation: str) -> int:
        """Perform arithmetic."""
        return 0

    agent = Agent(name="test_agent")
    agent.with_tools([search, calculator])
    stubs = agent._generate_type_stubs()

    assert "def search" in stubs
    assert "def calculator" in stubs
    assert "query" in stubs
    assert "operation" in stubs


def test_generate_type_stubs_preserves_descriptions():
    """Test that tool descriptions appear in the stubs."""
    def custom_tool(data: str) -> dict:
        """Process custom data with special handling."""
        return {}

    agent = Agent(name="test_agent")
    agent.with_tools([custom_tool])
    stubs = agent._generate_type_stubs()

    assert "Process custom data with special handling" in stubs


# ─── Codemode Prompt Tests ───


def test_build_codemode_prompt():
    """Test the codemode prompt includes type stubs and user input."""
    def search(query: str) -> str:
        """Search for things."""
        return query

    agent = Agent(name="test_agent")
    agent.with_tools([search])
    prompt = agent._build_codemode_prompt("find laptops")

    assert "class tools:" in prompt
    assert "def search" in prompt
    assert "find laptops" in prompt
    assert "tools.tool_name" in prompt
    assert "print()" in prompt


def test_build_codemode_prompt_includes_context():
    """Test that agent context is included in the codemode prompt."""
    agent = Agent(name="test_agent")
    agent.context = "You are a data analyst."

    def analyze(data: str) -> str:
        """Analyze data."""
        return data

    agent.with_tools([analyze])
    prompt = agent._build_codemode_prompt("analyze sales")

    assert "You are a data analyst." in prompt


# ─── Code Execution Tests ───


@pytest.mark.asyncio
async def test_exec_codemode_basic():
    """Test basic code execution with tools bridge."""
    call_log = []

    def search(query: str) -> str:
        """Search."""
        call_log.append(query)
        return f"Results for: {query}"

    agent = Agent(name="test_agent")
    agent.with_tools([search])

    code = 'result = tools.search(query="laptops")\nprint(result)'
    output = await agent._exec_codemode(code)

    assert "Results for: laptops" in output
    assert call_log == ["laptops"]


@pytest.mark.asyncio
async def test_exec_codemode_chained_calls():
    """Test chaining multiple tool calls in code."""
    def search(query: str) -> str:
        """Search."""
        return f"found: {query}"

    def save(content: str, filename: str) -> str:
        """Save to file."""
        return f"saved {filename}"

    agent = Agent(name="test_agent")
    agent.with_tools([search, save])

    code = '''result = tools.search(query="data")
saved = tools.save(content=result, filename="out.txt")
print(saved)'''
    output = await agent._exec_codemode(code)

    assert "saved out.txt" in output


@pytest.mark.asyncio
async def test_exec_codemode_error_handling():
    """Test that code errors are captured gracefully."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    code = "x = 1 / 0"  # ZeroDivisionError
    output = await agent._exec_codemode(code)

    assert "Error" in output
    assert "ZeroDivisionError" in output


@pytest.mark.asyncio
async def test_exec_codemode_syntax_error():
    """Test that syntax errors are captured."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    code = "def foo(:\n  pass"  # SyntaxError
    output = await agent._exec_codemode(code)

    assert "Error" in output
    assert "SyntaxError" in output


@pytest.mark.asyncio
async def test_exec_codemode_no_output():
    """Test code that produces no output."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    code = "x = 42"  # No print
    output = await agent._exec_codemode(code)

    assert output == "(no output)"


@pytest.mark.asyncio
async def test_exec_codemode_blocked_imports():
    """Test that dangerous modules are blocked."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    # os should be blocked
    code = "import os\nprint(os.getcwd())"
    output = await agent._exec_codemode(code)
    assert "Error" in output
    assert "not allowed" in output

    # subprocess should be blocked
    code = "import subprocess\nsubprocess.run(['ls'])"
    output = await agent._exec_codemode(code)
    assert "Error" in output


@pytest.mark.asyncio
async def test_exec_codemode_safe_imports():
    """Test that safe standard library modules can be imported."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    # math should work
    code = "import math\nprint(math.sqrt(16))"
    output = await agent._exec_codemode(code)
    assert "4.0" in output

    # json should work
    code = 'import json\ndata = json.dumps({"a": 1})\nprint(data)'
    output = await agent._exec_codemode(code)
    assert '"a"' in output


@pytest.mark.asyncio
async def test_exec_codemode_multiple_prints():
    """Test that multiple print outputs are captured."""
    agent = Agent(name="test_agent")

    def dummy(x: int) -> int:
        """Dummy."""
        return x

    agent.with_tools([dummy])

    code = 'print("line1")\nprint("line2")\nprint("line3")'
    output = await agent._exec_codemode(code)

    assert "line1" in output
    assert "line2" in output
    assert "line3" in output


@pytest.mark.asyncio
async def test_exec_codemode_with_loops():
    """Test code with loops and variables."""
    agent = Agent(name="test_agent")

    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.with_tools([add])

    code = '''total = 0
for i in range(5):
    total = tools.add(x=total, y=i)
print(f"total={total}")'''
    output = await agent._exec_codemode(code)

    assert "total=10" in output


# ─── Infer Codemode Integration Tests ───


@pytest.mark.asyncio
async def test_infer_codemode_agent_flag():
    """Test codemode=True on Agent enables code mode in infer()."""
    def search(query: str) -> str:
        """Search."""
        return f"found: {query}"

    agent = Agent(name="test_agent", codemode=True, enable_memory=False)
    agent.with_tools([search])

    mock_llm_response = '''```python
result = tools.search(query="test")
print(result)
```'''

    with patch.object(agent, '_call_llm', new_callable=AsyncMock, return_value=mock_llm_response):
        result = await agent.infer("search for test")

    assert result["codemode"] is True
    assert result["tool_used"] == "codemode"
    assert "found: test" in result["result"]


@pytest.mark.asyncio
async def test_infer_codemode_strips_markdown_fences():
    """Test that code is extracted from markdown fences."""
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    agent = Agent(name="test_agent", codemode=True, enable_memory=False)
    agent.with_tools([add])

    mock_llm_response = '''```python
result = tools.add(x=3, y=4)
print(result)
```'''

    with patch.object(agent, '_call_llm', new_callable=AsyncMock, return_value=mock_llm_response):
        result = await agent.infer("add 3 and 4")

    assert "7" in result["result"]
    # Code should not contain markdown fences
    assert "```" not in result["parameters"]["code"]


@pytest.mark.asyncio
async def test_infer_codemode_no_fences():
    """Test that raw code without fences also works."""
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    agent = Agent(name="test_agent", codemode=True, enable_memory=False)
    agent.with_tools([add])

    mock_llm_response = 'result = tools.add(x=5, y=6)\nprint(result)'

    with patch.object(agent, '_call_llm', new_callable=AsyncMock, return_value=mock_llm_response):
        result = await agent.infer("add 5 and 6")

    assert "11" in result["result"]


@pytest.mark.asyncio
async def test_infer_codemode_with_sandbox():
    """Test codemode works when set via with_sandbox."""
    def multiply(x: int, y: int) -> int:
        """Multiply numbers."""
        return x * y

    agent = Agent(name="test_agent", enable_memory=False)
    agent.with_sandbox(read_tools=[multiply], codemode=True)

    mock_llm_response = '''```python
result = tools.multiply(x=6, y=7)
print(result)
```'''

    with patch.object(agent, '_call_llm', new_callable=AsyncMock, return_value=mock_llm_response):
        result = await agent.infer("multiply 6 by 7")

    assert result["codemode"] is True
    assert "42" in result["result"]


@pytest.mark.asyncio
async def test_infer_codemode_auto_retry():
    """Test auto-retry when code execution fails."""
    def add(x: int, y: int) -> int:
        """Add numbers."""
        return x + y

    agent = Agent(name="test_agent", codemode=True, enable_memory=False)
    agent.with_tools([add])

    # First response has a bug, second is fixed
    call_count = 0
    async def mock_llm(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 'result = tools.add(x=3, y="oops")\nprint(result)'  # TypeError
        else:
            return 'result = tools.add(x=3, y=4)\nprint(result)'  # Fixed

    with patch.object(agent, '_call_llm', side_effect=mock_llm):
        result = await agent.infer("add 3 and 4")

    assert "7" in result["result"]
    assert result["attempts"] == 2  # Took 2 attempts


@pytest.mark.asyncio
async def test_infer_codemode_retry_reports_attempts():
    """Test that attempt count is tracked in response."""
    def greet(name: str) -> str:
        """Greet."""
        return f"Hi {name}"

    agent = Agent(name="test_agent", codemode=True, enable_memory=False)
    agent.with_tools([greet])

    # Success on first try
    mock_response = 'result = tools.greet(name="World")\nprint(result)'

    with patch.object(agent, '_call_llm', new_callable=AsyncMock, return_value=mock_response):
        result = await agent.infer("greet world")

    assert result["attempts"] == 1  # First try succeeded

