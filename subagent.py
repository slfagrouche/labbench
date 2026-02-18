"""Backward-compatibility shim for LabBench.

The legacy multi-agent subsystem is intentionally not included in LabBench.
This module keeps old imports from crashing with an opaque ImportError and
returns a clear message to migrate to skill-based workflows.
"""

from dataclasses import dataclass
from typing import Any


_ERR = (
    "Sub-agent APIs are not available in LabBench. "
    "Use built-in or custom skills instead."
)


@dataclass
class AgentDefinition:
    name: str = "unavailable"
    description: str = _ERR


@dataclass
class SubAgentTask:
    task_id: str = "unavailable"
    status: str = "error"
    result: str = _ERR


class SubAgentManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(_ERR)


def load_agent_definitions(*_args: Any, **_kwargs: Any):
    return []


def get_agent_definition(*_args: Any, **_kwargs: Any):
    return None


def _extract_final_text(*_args: Any, **_kwargs: Any) -> str:
    return _ERR


def _agent_run(*_args: Any, **_kwargs: Any) -> str:
    raise RuntimeError(_ERR)


_BUILTIN_AGENTS: list[AgentDefinition] = []
