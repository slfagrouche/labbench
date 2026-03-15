# Architecture Guide

This document is for developers who want to understand, modify, or extend LabBench.
For user-facing docs, see [README.md](../README.md).

---

## Overview

LabBench is a Python terminal assistant that lets LLMs (Claude, GPT, Gemini, local models, etc.)
drive tools, memory, and skills with a focus on notebooks and data workflows. The
codebase uses a flat module layout for readability.

```
User Input
    │
    ▼
labbench.py  ── REPL, slash commands, rendering
    │
    ├──► agent.py  ── multi-turn loop, permission gates
    │       │
    │       ├──► providers.py  ── API streaming (Anthropic / OpenAI-compat)
    │       ├──► tool_registry.py ──► tools.py  ── built-ins + memory/skill tools
    │       ├──► compaction.py  ── context window management
    │
    ├──► context.py  ── system prompt (git, CLAUDE.md, memory)
    │       └──► memory/  ── persistent file-based memory
    │
    ├──► skill/  ── markdown skills (loader, builtins, tools, executor)
    └──► config.py  ── configuration persistence
```

**Key invariant:** Dependencies flow downward. No circular imports at the module level.

---

## Module Reference

### `tool_registry.py` — Tool Registry

The central registry that all tools register into. This is the foundation for extensibility.

**Data model:**

```python
@dataclass
class ToolDef:
    name: str               # unique identifier (e.g. "Read", "MemorySave")
    schema: dict            # JSON schema sent to the LLM API
    func: Callable          # (params: dict, config: dict) -> str
    read_only: bool         # True = auto-approve in 'auto' permission mode
    concurrent_safe: bool   # True = safe to run in parallel
```

**Public API:**

| Function | Description |
|---|---|
| `register_tool(tool_def)` | Add a tool to the registry (overwrites by name) |
| `get_tool(name)` | Look up by name, returns `None` if not found |
| `get_all_tools()` | List all registered tools |
| `get_tool_schemas()` | Return schemas for API calls |
| `execute_tool(name, params, config, max_output=32000)` | Execute with output truncation |
| `clear_registry()` | Reset — for testing only |

**Output truncation:** If a tool returns more than `max_output` chars, the result is
truncated to `first_half + [... N chars truncated ...] + last_quarter`. This prevents
a single tool call (e.g. reading a huge file) from blowing up the context window.

**Registering a custom tool:**

```python
from tool_registry import ToolDef, register_tool

def my_tool(params, config):
    return f"Hello, {params['name']}!"

register_tool(ToolDef(
    name="MyTool",
    schema={
        "name": "MyTool",
        "description": "A greeting tool",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    func=my_tool,
    read_only=True,
    concurrent_safe=True,
))
```

### `tools.py` — Built-in Tool Implementations

Contains core built-in tools (Read/Write/Edit/Bash/Glob/Grep/WebFetch/WebSearch/NotebookEdit/GetDiagnostics/AskUserQuestion),
then imports `memory.tools` and `skill.tools` so memory/skill tools auto-register via `tool_registry`.

**Key internals:**

- `_is_safe_bash(cmd)` — whitelist of safe shell commands for auto-approval
- `generate_unified_diff(old, new, filename)` — diff generation for Edit/Write
- `maybe_truncate_diff(diff_text, max_lines=80)` — truncate large diffs for display
- Backward-compatible `execute_tool(name, inputs, permission_mode, ask_permission)` wrapper

### `agent.py` — Core Agent Loop

The heart of the system. `run()` is a generator that yields events as they happen.

```python
def run(user_message, state, config, system_prompt,
        depth=0, cancel_check=None) -> Generator:
```

**Loop logic:**

```
1. Append user message
2. Inject depth metadata into config
3. While True:
   a. Check cancel_check() (if provided)
   b. maybe_compact(state, config) — compress if near context limit
   c. Stream from provider → yield TextChunk / ThinkingChunk
   d. Record assistant message
   e. If no tool_calls → break
   f. For each tool_call:
      - Permission check (_check_permission)
      - If denied → yield PermissionRequest → user decides
      - Execute tool → yield ToolStart / ToolEnd
      - Append tool result
   g. Loop (model sees tool results and responds)
```

**Event types:**

| Event | Fields | When |
|---|---|---|
| `TextChunk` | `text` | Streaming text delta |
| `ThinkingChunk` | `text` | Extended thinking block |
| `ToolStart` | `name, inputs` | Before tool execution |
| `ToolEnd` | `name, result, permitted` | After tool execution |
| `PermissionRequest` | `description, granted` | Needs user approval |
| `TurnDone` | `input_tokens, output_tokens` | End of one API turn |

### `compaction.py` — Context Window Management

Keeps conversations within model context limits using two layers.

**Layer 1: Snip** (`snip_old_tool_results`)
- Rule-based, no API cost
- Truncates tool-role messages older than `preserve_last_n_turns` (default 6)
- Keeps first half + last quarter of the content

**Layer 2: Auto-Compact** (`compact_messages`)
- Model-driven: calls the current model to summarize old messages
- Splits messages into [old | recent] at ~70/30 ratio
- Replaces old messages with a summary + acknowledgment

**Trigger:** `maybe_compact()` checks `estimate_tokens(messages) > context_limit * 0.7`.
Runs snip first (cheap), then auto-compact if still over.

**Token estimation:** `len(content) / 3.5` — simple heuristic. Works for most models.
`get_context_limit(model)` reads from the provider registry.

### `memory/` — Persistent Memory

File-based memory system stored in `~/.labbench/memory/`.

**Storage format:**

```
~/.labbench/memory/
├── MEMORY.md              # Index: one line per memory
├── user_preferences.md    # Individual memory file
└── project_auth.md
```

Each memory file uses markdown with YAML frontmatter:

```markdown
---
name: user preferences
description: coding style preferences
type: feedback
created: 2026-04-02
---

User prefers 4-space indentation and type hints.
```

**How it integrates:**
- `get_memory_context()` returns the MEMORY.md index text
- `context.py` injects this into the system prompt
- The model reads the index, then uses `Read` tool to access full memory content
- The model uses `MemorySave` / `MemoryDelete` tools to manage memories

### Sub-agent note

LabBench does not ship the older threaded sub-agent subsystem in this branch. Keep architecture
and contributor docs focused on the active core (`labbench.py`, `agent.py`, `tools.py`, `memory/`, `skill/`).

### `skill/` — Reusable prompt templates

The **`skill`** package loads markdown skills with YAML frontmatter. They are **not code** — structured prompts injected into the agent loop. Built-ins live in `skill/builtin.py`; user/project skills live under `~/.labbench/skills/` and `./.labbench/skills/`.

**Skill file format:**

```markdown
---
name: commit
description: Create a conventional commit
triggers: ["/commit"]
tools: [Bash, Read]
---

Your prompt instructions here...
```

**Execution:** `execute_skill()` wraps the skill prompt as a user message and calls
`agent.run()`. The skill runs through the exact same agent loop as a normal query.

**Search order:** Project-level (`./.labbench/skills/`) overrides user-level
(`~/.labbench/skills/`) when skill names collide.

### `providers.py` — Multi-Provider Abstraction

Two streaming adapters cover all providers:

| Adapter | Providers |
|---|---|
| `stream_anthropic()` | Anthropic (native SDK) |
| `stream_openai_compat()` | OpenAI, Gemini, Kimi, Qwen, Zhipu, DeepSeek, Ollama, LM Studio, Custom |

**Neutral message format** (provider-independent):

```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "name": "...", "input": {...}}]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

Conversion functions: `messages_to_anthropic()`, `messages_to_openai()`, `tools_to_openai()`.

**Provider-specific handling:**
- Gemini 3 models require `thought_signature` in tool call responses — this is transparently
  captured and passed through via `extra_content` on tool_call dicts.

### `context.py` — System Prompt Builder

Assembles the system prompt from:
1. Base template (role, date, cwd, platform)
2. Git info (branch, status, recent commits)
3. CLAUDE.md content (project-level + global)
4. Memory index (from `memory.get_memory_context()`)

### `config.py` — Configuration

Defaults stored in `~/.labbench/config.json`. Key settings:

| Key | Default | Description |
|---|---|---|
| `model` | `claude-opus-4-6` | Active model |
| `max_tokens` | `8192` | Max output tokens |
| `permission_mode` | `auto` | Permission mode |
| `max_tool_output` | `32000` | Tool output truncation limit |
| `max_agent_depth` | `3` | Max nested agent depth |
| `max_concurrent_agents` | `3` | Thread pool size |

---

## Data Flow Example

A user asks "Read config.py and change max_tokens to 16384":

```
1. labbench.py captures input
2. agent.run() appends user message, calls maybe_compact()
3. providers.stream() sends to provider API with the currently registered tool schemas
4. Model responds: text + tool_call[Read(config.py)]
5. agent.py checks permission (Read = read_only → auto-approve)
6. tool_registry.execute_tool("Read", ...) → file content (truncated if >32K)
7. Tool result appended to messages, loop back to step 3
8. Model responds: text + tool_call[Edit(config.py, "8192", "16384")]
9. agent.py checks permission (Edit = not read_only → ask user)
10. User approves → tools.py._edit() runs, generates diff
11. labbench.py renders diff with ANSI colors (red/green)
12. Tool result appended, loop back to step 3
13. Model responds: "Done, max_tokens changed to 16384"
14. No tool_calls → loop ends, TurnDone yielded
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_tool_registry.py -v
python -m pytest tests/test_compaction.py -v
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_skills.py -v
python -m pytest tests/test_diff_view.py -v
```

Tests use `monkeypatch` and `tmp_path` fixtures to avoid side effects.

