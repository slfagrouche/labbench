"""File-based memory storage with user-level and project-level scopes.

Storage layout:
  user scope    : ~/.labbench/memory/<slug>.md
  project scope : .labbench/memory/<slug>.md  (relative to cwd)

MEMORY.md in each directory is the index file — rebuilt automatically after
every save/delete. It is loaded into the system prompt to give the model an
overview of available memories.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────

USER_MEMORY_DIR = Path.home() / ".labbench" / "memory"
INDEX_FILENAME = "MEMORY.md"

# Maximum lines/bytes for the index file
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000


def get_project_memory_dir() -> Path:
    """Return the project-local memory directory (relative to cwd)."""
    return Path.cwd() / ".labbench" / "memory"


def get_memory_dir(scope: str = "user") -> Path:
    """Return the memory directory for the given scope.

    Args:
        scope: "user" (global ~/.labbench/memory) or
               "project" (.labbench/memory relative to cwd)
    """
    if scope == "project":
        return get_project_memory_dir()
    return USER_MEMORY_DIR


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry loaded from a .md file.

    Attributes:
        name:        human-readable name (also the display title in the index)
        description: short one-line description (used for relevance decisions)
        type:        "user" | "feedback" | "project" | "reference"
        content:     body text of the memory
        file_path:   absolute path to the .md file on disk
        created:     date string, e.g. "2026-04-02"
        scope:       "user" | "project" — which directory this was loaded from
    """
    name: str
    description: str
    type: str
    content: str
    file_path: str = ""
    created: str = ""
    scope: str = "user"


# ── Helpers ────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert name to a filesystem-safe slug (max 60 chars)."""
    s = name.lower().strip().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s[:60]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse ---\\nkey: value\\n---\\nbody format.

    Returns:
        (meta_dict, body_str)
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta, parts[2].strip()


def _format_entry_md(entry: MemoryEntry) -> str:
    """Render a MemoryEntry as a markdown file with YAML frontmatter."""
    return (
        f"---\n"
