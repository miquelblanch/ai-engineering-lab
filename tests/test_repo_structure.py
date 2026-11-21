"""Repository conventions, checked in CI so they don't drift silently.

Every project folder needs a README that states the question it answers,
and the two write-up templates must keep the sections the site relies on.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = sorted(p for p in (ROOT / "projects").iterdir() if p.is_dir())


def test_there_are_projects():
    assert PROJECTS, "projects/ is empty"


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_project_readme_states_its_question(project: Path):
    readme = project / "README.md"
    assert readme.is_file(), f"{project.name} has no README.md"
    assert "**Question:**" in readme.read_text(encoding="utf-8")


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_project_is_listed_in_root_readme(project: Path):
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"projects/{project.name}/" in root_readme


@pytest.mark.parametrize(
    ("template", "sections"),
    [
        ("case-study.md", ["## Hypothesis", "## Results", "## What is still unknown"]),
        ("learning-log.md", ["## What I built", "## Alternatives ruled out", "## Hours"]),
    ],
)
def test_templates_keep_their_sections(template: str, sections: list[str]):
    text = (ROOT / "templates" / template).read_text(encoding="utf-8")
    missing = [s for s in sections if s not in text]
    assert not missing, f"{template} lost sections: {missing}"
