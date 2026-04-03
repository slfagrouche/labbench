from __future__ import annotations

import pytest
from pathlib import Path

import skill.loader as _loader
from skill.loader import _parse_skill_file, _parse_list_field, find_skill, SkillDef
from skill import load_skills, substitute_arguments


COMMIT_MD = """\
---
name: commit
description: Create a git commit
triggers: [/commit, commit changes]
tools: [Bash, Read]
---
Review staged changes and create a commit with a descriptive message.
"""

REVIEW_MD = """\
---
name: review
description: Review a pull request
triggers: [/review, /review-pr]
tools: [Bash, Read, Grep]
---
Analyze the PR diff and provide constructive feedback.
"""

ARGS_MD = """\
---
name: deploy
description: Deploy to an environment
triggers: [/deploy]
tools: [Bash]
argument-hint: [env] [version]
arguments: [env, version]
---
Deploy $VERSION to $ENV environment. Full args: $ARGUMENTS
"""


@pytest.fixture()
def skill_dir(tmp_path, monkeypatch):
    """Create a temp skill directory with sample skills and patch _get_skill_paths."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "commit.md").write_text(COMMIT_MD, encoding="utf-8")
    (skills_dir / "review.md").write_text(REVIEW_MD, encoding="utf-8")

    monkeypatch.setattr(_loader, "_get_skill_paths", lambda: [skills_dir])
    # Also patch the builtin list to be empty so tests are predictable
    monkeypatch.setattr(_loader, "_BUILTIN_SKILLS", [])
    return skills_dir


# ------------------------------------------------------------------
# _parse_list_field
# ------------------------------------------------------------------

def test_parse_list_field_bracket():
    assert _parse_list_field("[a, b, c]") == ["a", "b", "c"]


def test_parse_list_field_plain():
    assert _parse_list_field("a, b, c") == ["a", "b", "c"]


def test_parse_list_field_single():
    assert _parse_list_field("solo") == ["solo"]


# ------------------------------------------------------------------
# _parse_skill_file
# ------------------------------------------------------------------

def test_parse_skill_file(skill_dir):
    path = skill_dir / "commit.md"
    skill = _parse_skill_file(path)
    assert skill is not None
    assert skill.name == "commit"
    assert skill.description == "Create a git commit"
    assert "/commit" in skill.triggers
    assert "commit changes" in skill.triggers
    assert "Bash" in skill.tools
    assert "Read" in skill.tools
    assert "commit" in skill.prompt.lower()
    assert skill.file_path == str(path)


def test_parse_skill_file_review(skill_dir):
    path = skill_dir / "review.md"
    skill = _parse_skill_file(path)
    assert skill is not None
    assert skill.name == "review"
    assert "/review" in skill.triggers
    assert "/review-pr" in skill.triggers


def test_parse_skill_file_invalid(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("no frontmatter here", encoding="utf-8")
    assert _parse_skill_file(bad) is None


def test_parse_skill_file_no_name(tmp_path):
    no_name = tmp_path / "noname.md"
    no_name.write_text("---\ndescription: test\n---\nbody\n", encoding="utf-8")
    assert _parse_skill_file(no_name) is None
