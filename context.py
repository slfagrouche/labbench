"""System context: CLAUDE.md, git info, cwd injection."""
import os
import subprocess
from pathlib import Path
from datetime import datetime

from memory import get_memory_context

SYSTEM_PROMPT_TEMPLATE = """\
You are **LabBench**, a small terminal AI assistant focused on Python, notebooks, and data work.
You help with editing, debugging, refactors, and analysis—prefer clear, minimal steps.

# Available Tools

## File & shell
- **Read**, **Write**, **Edit**, **Bash**, **Glob**, **Grep**
- **WebFetch**, **WebSearch**
- **NotebookEdit** (Jupyter `.ipynb`), **GetDiagnostics** (linters) when relevant

## Memory
- **MemorySave**, **MemoryDelete**, **MemorySearch**, **MemoryList**

## Skills
- **Skill**, **SkillList** — reusable prompt templates; `/skills` in the REPL lists them.

## Interaction
- **AskUserQuestion** — ask the user a clarifying question with optional choices.

# Guidelines
- Be concise. Lead with the answer.
- For notebooks, keep cells reproducible and outputs reasonable in size.
- Prefer editing existing files over creating new ones.
- Use absolute paths for file operations.
- If scope is unclear, ask before large refactors.

# Environment
- Current date: {date}
- Working directory: {cwd}
- Platform: {platform}
{git_info}{claude_md}"""


def get_git_info() -> str:
    """Return git branch/status summary if in a git repo."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL, text=True).strip()
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            stderr=subprocess.DEVNULL, text=True).strip()
        parts = [f"- Git branch: {branch}"]
        if status:
            lines = status.split('\n')[:10]
            parts.append("- Git status:\n" + "\n".join(f"  {l}" for l in lines))
        if log:
            parts.append("- Recent commits:\n" + "\n".join(f"  {l}" for l in log.split('\n')))
        return "\n".join(parts) + "\n"
    except Exception:
        return ""


def get_claude_md() -> str:
    """Load CLAUDE.md from cwd or parents, and ~/.claude/CLAUDE.md."""
    content_parts = []

    # Global CLAUDE.md
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if global_md.exists():
        try:
            content_parts.append(f"[Global CLAUDE.md]\n{global_md.read_text()}")
        except Exception:
            pass

    # Project CLAUDE.md (walk up from cwd)
    p = Path.cwd()
    for _ in range(10):
        candidate = p / "CLAUDE.md"
        if candidate.exists():
            try:
                content_parts.append(f"[Project CLAUDE.md: {candidate}]\n{candidate.read_text()}")
            except Exception:
                pass
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    if not content_parts:
        return ""
    return "\n# Memory / CLAUDE.md\n" + "\n\n".join(content_parts) + "\n"


def build_system_prompt() -> str:
    import platform
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %A"),
        cwd=str(Path.cwd()),
        platform=platform.system(),
        git_info=get_git_info(),
        claude_md=get_claude_md(),
    )
    memory_ctx = get_memory_context()
    if memory_ctx:
        prompt += f"\n\n# Memory\nYour persistent memories:\n{memory_ctx}\n"
    return prompt
