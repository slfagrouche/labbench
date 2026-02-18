"""Built-in skills that ship with LabBench."""
from __future__ import annotations

from .loader import SkillDef, register_builtin_skill

# ── /commit ────────────────────────────────────────────────────────────────

_COMMIT_PROMPT = """\
Review the current git state and create a well-structured commit.

## Steps

1. Run `git status` and `git diff --staged` to see what is staged.
   - If nothing is staged, run `git diff` to see unstaged changes, then stage relevant files.
2. Analyze the changes:
   - Summarize the nature of the change (feature, bug fix, refactor, docs, etc.)
   - Write a concise commit title (≤72 chars) focusing on *why*, not just *what*.
   - If multiple logical changes exist, ask the user whether to split them.
3. Create the commit:
   ```
   git commit -m "<title>"
   ```
   If additional context is needed, add a body separated by a blank line.
4. Print the commit hash and summary when done.

**Rules:**
- Never use `--no-verify`.
- Never commit files that likely contain secrets (.env, credentials, keys).
- Prefer imperative mood in the title: "Add X", "Fix Y", "Refactor Z".

User context: $ARGUMENTS
"""

_REVIEW_PROMPT = """\
Review the code or pull request and provide structured feedback.

## Steps

1. Understand the scope:
   - If a PR number or URL is given in $ARGUMENTS, use `gh pr view $ARGUMENTS --patch` to get the diff.
   - Otherwise, use `git diff main...HEAD` (or `git diff HEAD~1`) for local changes.
2. Analyze the diff:
   - Correctness: Are there bugs, edge cases, or logic errors?
   - Security: Injection, auth issues, exposed secrets, unsafe operations?
   - Performance: N+1 queries, unnecessary allocations, blocking calls?
   - Style: Does it follow existing conventions in the codebase?
   - Tests: Are new behaviors tested? Do existing tests cover the change?
3. Write a structured review:
   ```
   ## Summary
   One-line overview of what the change does.

   ## Issues
   - [CRITICAL/MAJOR/MINOR] Description and location

   ## Suggestions
   - Nice-to-have improvements

   ## Verdict
   APPROVE / REQUEST CHANGES / COMMENT
   ```
4. If changes are needed, list specific file:line references.

User context: $ARGUMENTS
"""

_EDA_PROMPT = """\
Help the user with **exploratory data analysis** or **notebook cleanup** in Python.

## Context from user
$ARGUMENTS

## Steps

1. **Scope:** Identify whether work targets a `.ipynb`, scripts, or `data/` files. Prefer reading notebooks with the notebook-aware tools when available.
2. **Data sanity:** Note dtypes, missing values, obvious outliers; suggest `head`, shape, and column lists before heavy compute.
3. **Reproducibility:** Encourage deterministic cell order, clear markdown section headers, and avoiding huge embedded outputs when not needed.
4. **Plots:** If visualizing, prefer clear labels, titles, and one chart per insight unless comparing.
5. **Next actions:** End with a short checklist: what to run next, what to verify, what could break.

Stay concise; use tools to inspect the repo rather than guessing paths.
"""


def _register_builtins() -> None:
    register_builtin_skill(SkillDef(
        name="commit",
        description="Review staged changes and create a well-structured git commit",
        triggers=["/commit"],
        tools=["Bash", "Read"],
        prompt=_COMMIT_PROMPT,
        file_path="<builtin>",
        when_to_use="Use when the user wants to commit changes. Triggers: '/commit', 'commit changes', 'make a commit'.",
        argument_hint="[optional context]",
        arguments=[],
        user_invocable=True,
        context="inline",
        source="builtin",
    ))

    register_builtin_skill(SkillDef(
        name="review",
        description="Review code changes or a pull request and provide structured feedback",
        triggers=["/review", "/review-pr"],
        tools=["Bash", "Read", "Grep"],
        prompt=_REVIEW_PROMPT,
        file_path="<builtin>",
        when_to_use="Use when the user wants a code review. Triggers: '/review', '/review-pr', 'review this PR'.",
        argument_hint="[PR number or URL]",
        arguments=["pr"],
        user_invocable=True,
        context="inline",
        source="builtin",
    ))

    register_builtin_skill(SkillDef(
        name="eda",
        description="Exploratory data analysis and notebook hygiene for Python data work",
        triggers=["/eda", "/notebook"],
        tools=["Read", "Glob", "Grep", "Bash"],
        prompt=_EDA_PROMPT,
        file_path="<builtin>",
        when_to_use="Use for notebooks, pandas workflows, or cleaning analysis code. Triggers: '/eda', '/notebook'.",
        argument_hint="[focus area or file path]",
        arguments=[],
        user_invocable=True,
        context="inline",
        source="builtin",
    ))


_register_builtins()
