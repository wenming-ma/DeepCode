# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCode is an AI-powered multi-agent system that automates code generation from research papers (Paper2Code), natural language descriptions (Text2Web/Text2Backend), and various document formats. It uses the Model Context Protocol (MCP) for tool integration and supports multiple LLM providers (OpenAI, Anthropic, Google).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Or with UV (recommended for development)
uv venv --python=3.13 && source .venv/bin/activate && uv pip install -r requirements.txt

# Launch web interface (choose one)
deepcode                              # Option 1: After pip install deepcode-hku
streamlit run ui/streamlit_app.py     # Option 2: From source code

# Launch CLI interface
python cli/main_cli.py

# Run linting
pre-commit run --all-files

# Test paper reproduction
python deepcode.py test <paper_name>
python deepcode.py test <paper_name> --fast  # Skip GitHub download and indexing
```

## Architecture

### Multi-Agent Orchestration Engine
The core system (`workflows/agent_orchestration_engine.py`) coordinates specialized AI agents through a sequential pipeline:

1. **ResearchAnalyzerAgent** - Analyzes input (papers, URLs, text) and extracts metadata
2. **ResourceProcessorAgent** - Downloads/moves files to workspace (`deepcode_lab/papers/<id>/`)
3. **DocumentSegmentationAgent** - Optional intelligent document chunking for large papers
4. **ConceptAnalysisAgent + AlgorithmAnalysisAgent** - Parallel analysis via ParallelLLM fan-out
5. **CodePlannerAgent** - Fan-in agent that synthesizes analysis into YAML implementation plan
6. **ReferenceAnalysisAgent** - Extracts GitHub repos from paper references
7. **GithubDownloadAgent** - Downloads reference repositories to `code_base/`
8. **CodeImplementationWorkflow** - Generates code from the plan

### Key Components

- **`workflows/`** - Agent definitions and workflow orchestration
  - `agent_orchestration_engine.py` - Main pipeline orchestrator with two entry points:
    - `execute_multi_agent_research_pipeline()` - Full paper-to-code pipeline
    - `execute_chat_based_planning_pipeline()` - Direct text-to-code (skips paper analysis)
  - `code_implementation_workflow.py` - Standard code generation workflow
  - `code_implementation_workflow_index.py` - Enhanced workflow with code reference indexing

- **`tools/`** - MCP server implementations (Python scripts that expose tools via MCP)
  - `code_implementation_server.py` - File ops, code execution, workspace management
  - `code_reference_indexer.py` - Intelligent code search from indexed repos
  - `document_segmentation_server.py` - Smart document chunking
  - `pdf_downloader.py` - File download/move with format conversion

- **`prompts/code_prompts.py`** - All agent system prompts (PAPER_INPUT_ANALYZER_PROMPT, CODE_PLANNING_PROMPT_TRADITIONAL, etc.)

- **`utils/llm_utils.py`** - LLM provider selection, token limit management, adaptive configuration

### Configuration

- **`mcp_agent.config.yaml`** - Main config: MCP server definitions, LLM provider selection (`llm_provider: "google"|"anthropic"|"openai"`), document segmentation settings
- **`mcp_agent.secrets.yaml`** - API keys (not committed)

### Workspace Structure

Generated during pipeline execution:
```
deepcode_lab/
  papers/
    <id>/
      <id>.md           # Converted paper content
      initial_plan.txt  # YAML implementation plan
      reference.txt     # Reference analysis
      code_base/        # Downloaded GitHub repos
      code/             # Generated implementation
```

## Code Style

- Uses `ruff` for formatting and linting (configured in `.pre-commit-config.yaml`)
- Async/await throughout for agent coordination
- Type hints with `Optional`, `Dict`, `List`, `Tuple`, `Callable` from typing
- Agents use `RequestParams(maxTokens=..., temperature=...)` for LLM calls

## Windows Notes

MCP servers may need absolute paths in `mcp_agent.config.yaml`:
```yaml
brave:
  command: "node"
  args: ["C:/path/to/node_modules/@modelcontextprotocol/server-brave-search/dist/index.js"]
```

## Custom Anthropic API Endpoint

To use a custom Anthropic-compatible API endpoint (e.g., proxy services), configure `mcp_agent.secrets.yaml`:

```yaml
anthropic:
  api_key: "your-api-key"
  base_url: "https://your-custom-endpoint.com/api"
```

### mcp-agent Library Bug Fix

The `mcp-agent` library has a bug where it doesn't pass `base_url` to the Anthropic client. You need to manually patch the library:

**File**: `.venv/Lib/site-packages/mcp_agent/workflows/llm/augmented_llm_anthropic.py`

**Location**: `request_completion_task` function (around line 809)

**Fix**: Change:
```python
client = AsyncAnthropic(api_key=request.config.api_key)
```

To:
```python
client = AsyncAnthropic(api_key=request.config.api_key, base_url=request.config.base_url)
```

This fix is required for custom Anthropic endpoints to work with the main agent orchestration pipeline. Without this fix, you'll get `401 authentication_error: invalid x-api-key` errors because the request goes to the official Anthropic API instead of your custom endpoint.
