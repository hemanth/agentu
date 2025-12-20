# Changelog

All notable changes to this project will be documented in this file.

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
