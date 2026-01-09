# Changelog

All notable changes to agentu will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2025-01-09

### Added
- **Ralph Mode**: Autonomous agent loop inspired by [ghuntley.com/ralph](https://ghuntley.com/ralph)
  - `agent.ralph()` method for continuous autonomous execution
  - Reads PROMPT.md file with goal and checkpoints
  - Automatic completion detection via checkbox parsing
  - Checkpoint saving every N iterations
  - Max iterations and timeout guards for safety
  - Progress callbacks for monitoring
  - State updates written back to PROMPT.md
- `RalphRunner`, `RalphConfig`, `RalphState` classes in `ralph.py`
- `ralph()` standalone function for simpler usage
- Example: `examples/ralph_demo.py`
- Test suite: `tests/test_ralph.py` (10 tests)

### Changed
- Bumped version to 1.6.0

## [1.5.0] - 2025-12-29

### Added
- **Real-time Observability Dashboard**: Minimalist black/white web UI at `/dashboard` endpoint
  - Live metrics display (tool calls, LLM requests, errors, duration)
  - Event stream with verbose metadata and local timestamps
  - Auto-refresh every second
  - Built with web components (`<metric-card>`, `<event-stream>`)
- **Auto-instrumentation**: Automatic tracking of all tool calls and LLM requests
  - `Observer` class in `observe.py` module
  - Event types: `tool_call`, `llm_request`, `inference_start/end`, `error`, `session_create/end`
  - Trace context manager for timing code blocks
- **Observability API**: 
  - `GET /dashboard` - Serves observability UI
  - `GET /api/metrics` - Returns metrics and events as JSON
- **Flexible Output Formats**:
  - Console: Color-coded output with symbols (🔧, 🤖, ✓, ✗)
  - JSON: Structured logs for parsing
  - Silent: Metrics only, no output
- **Global Configuration**: `observe.configure(output='console'|'json'|'silent', enabled=True)`
- Dashboard integrated into existing FastAPI `serve()` - no separate server needed

### Changed
- Standardized branding from "AgentU" to lowercase "agentu" throughout codebase
- Event timestamps now use local time instead of UTC
- Dashboard HTML moved to `src/agentu/static/` folder for cleaner organization
- Improved event display with tool names, parameters, and query details

### Fixed
- EventType handling now works with both enum and string types
- Metrics update correctly for both EventType enums and string event types

## [1.4.0] - 2025-12-29

### Added
- **Agent Evaluation System**: Test and validate agent responses
  - `evaluate()` function for running test cases
  - Multiple matching strategies: exact, substring, LLM-judge, custom validators
  - `EvalResult` dataclass with pass rate, failed cases, and timing
  - Color-coded console output (✓ green, ✗ red)
  - JSON export via `result.to_json()` and `result.to_dict()`
- Comprehensive test coverage for evaluation features
- Example script demonstrating all evaluation capabilities

### Changed
- Version bumped to 1.4.0
- Added `evaluate`, `EvalResult`, `FailedCase` to package exports

## [1.3.0] - 2025-12-29

### Added
- **Stateful Sessions**: Session-based conversational memory
  - `Session` class for managing agent state across multiple interactions
  - `SessionManager` for handling multiple sessions
  - Automatic conversation history tracking
  - Session persistence and restoration
  - Integration with agent memory system
- Example scripts for session usage
- Comprehensive test coverage for session features

### Changed
- Version bumped to 1.3.0
- Added `Session`, `SessionManager` to package exports

## [1.2.0] - 2024-XX-XX

### Added
- MCP (Model Context Protocol) support
- Workflow system for multi-step agent tasks
- Skills system for reusable agent capabilities
- Memory storage backends (JSON, SQLite)

### Changed
- Improved tool parameter handling
- Enhanced error messages

## [1.1.0] - 2024-XX-XX

### Added
- Memory system with short-term and long-term storage
- Vector similarity search for memory retrieval
- Memory importance scoring

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- Agent class for creating AI agents
- Tool system for custom functions
- REST API server with FastAPI
- Basic examples and documentation

[1.5.0]: https://github.com/hemanth/agentu/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/hemanth/agentu/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/hemanth/agentu/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/hemanth/agentu/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/hemanth/agentu/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/hemanth/agentu/releases/tag/v1.0.0

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-01-09

### Added
- **Auto-detection of Ollama models** via `/api/tags` endpoint
- `get_ollama_models()` helper function to fetch available models
- `get_default_model()` helper function to select best available model
- Comprehensive tests for dynamic model detection (7 new tests)

### Changed
- Agent default model now auto-detects from Ollama instead of hardcoding
- Default fallback model changed from `llama2` to `qwen3:latest`
- `SearchAgent` now uses auto-detection instead of hardcoded model
- Agent `model` parameter now defaults to `None` (auto-detect)
- Updated README to highlight auto-detection feature
- All test fixtures updated to use `qwen3:latest`

### Fixed
- Agent initialization now properly detects available models from Ollama
- Graceful fallback to `qwen3:latest` when Ollama is unreachable

## [1.2.0] - 2025-01-20

### Added
- **Skills System**: Progressive loading skills that reduce context by 96%+
  - `Skill` class with 3-level loading (metadata, instructions, resources)
  - `.with_skills()` chainable API for attaching skills to agents
  - Auto-loading: Skills activate automatically based on user prompts
  - `get_skill_resource` tool for on-demand resource access
  - Works with any LLM (Ollama, OpenAI, vLLM, etc.)
- Example PDF processing skill with resources
- 24 new unit tests for Skills functionality

### Changed
- Enhanced `infer()` to auto-detect and activate relevant skills
- Updated context building to include skill instructions when triggered

## [1.1.0] - 2024-12-19

### Added
- **Workflow system** with operator-based composition (`>>` for sequential, `&` for parallel)
- `Agent.__call__()` method to create workflow steps with clean syntax
- Comprehensive workflow tests (18 new tests)
- `workflow.py` module with `Step`, `SequentialStep`, and `ParallelStep` classes
- Automatic context passing between workflow steps
- Lambda support for precise data flow control in workflows
- New workflow examples (`examples/workflow.py`, `examples/orchestrator.py`)

### Changed
- **BREAKING**: Removed `Orchestrator`, `ExecutionMode`, `Task`, and `Message` classes
- **BREAKING**: Removed `add_tool()`, `add_tools()`, `add_agent()`, `add_agents()` methods
- **BREAKING**: Removed `execute_tool()` and `process_input()` (use `call()` and `infer()`)
- Simplified API: `with_tools()` and `with_agents()` now always require lists
- Updated all examples to use new workflow operators
- Simplified README with real-world automated code review example
- Updated MCP implementation to use `with_mcp()` method
- Changed tagline to "The sleekest way to build AI agents"
- Updated default model examples from `llama3` to `qwen3`

### Removed
- Orchestrator-based multi-agent system (replaced by workflow operators)
- Task class for simple use cases (still available for advanced scenarios)
- Backward compatibility aliases
- `SERVING.md` (documentation consolidated into README)
- Redundant example files

### Fixed
- MCP configuration to use correct `type` parameter instead of `auth_type`
- Agent initialization to use `with_mcp()` instead of removed `load_mcp_tools()`
- All orchestrator references updated to use `infer()` instead of `process_input()`
- Test compatibility with new workflow system

## [1.0.0] - 2025-01-09

### Added
- **Workflow system** with operator-based composition (`>>` for sequential, `&` for parallel)
- `Agent.__call__()` method to create workflow steps with clean syntax
- Comprehensive workflow tests (18 new tests)
- `workflow.py` module with `Step`, `SequentialStep`, and `ParallelStep` classes
- Automatic context passing between workflow steps
- Lambda support for precise data flow control in workflows
- New workflow examples (`examples/workflow.py`, `examples/orchestrator.py`)

### Changed
- **BREAKING**: Removed `Orchestrator`, `ExecutionMode`, `Task`, and `Message` classes
- **BREAKING**: Removed `add_tool()`, `add_tools()`, `add_agent()`, `add_agents()` methods
- **BREAKING**: Removed `execute_tool()` and `process_input()` (use `call()` and `infer()`)
- Simplified API: `with_tools()` and `with_agents()` now always require lists
- Updated all examples to use new workflow operators
- Simplified README with real-world automated code review example
- Updated MCP implementation to use `with_mcp()` method
- Changed tagline to "The sleekest way to build AI agents"
- Updated default model examples from `llama3` to `qwen3`

### Removed
- Orchestrator-based multi-agent system (replaced by workflow operators)
- Task class for simple use cases (still available for advanced scenarios)
- Backward compatibility aliases
- `SERVING.md` (documentation consolidated into README)
- Redundant example files

### Fixed
- MCP configuration to use correct `type` parameter instead of `auth_type`
- Agent initialization to use `with_mcp()` instead of removed `load_mcp_tools()`
- All orchestrator references updated to use `infer()` instead of `process_input()`
- Test compatibility with new workflow system

## [0.3.0] - Previous release

Initial release with basic agent functionality, tools, memory, and orchestration.
