"""Split repositories into line-ranged chunks that can be cited as file:line links.

Everything indexed here can end up quoted on a public page, so this module is
deliberately conservative: it skips secrets, env files, build output and user data.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from app.models import Chunk

INCLUDE_SUFFIXES = {
    ".py", ".java", ".ts", ".tsx", ".js", ".md", ".sql", ".tf", ".yml", ".yaml", ".toml",
}
INCLUDE_NAMES = {"Dockerfile", "Dockerfile.lambda", "docker-compose.yml"}
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "target", "dist", "build", "__pycache__",
    ".mvn", ".pytest_cache", ".idea", ".vscode", "uploads", "generated_cvs", "data", ".claude",
    "evals", "fixtures",
    "profile",  # my CV/profile is indexed separately as self-reported claims, never as code
}
MAX_FILE_BYTES = 300_000

# Files that QUOTE requirements (sample JDs, prompts, eval data) are not evidence of
# having the skill. Put this marker near the top of such a file to keep it out.
NO_INDEX_MARKER = "fitcheck: no-index"

# A chunk that looks like it contains a credential is never indexed.
SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}"
    r"|nvapi-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"])",
    re.IGNORECASE,
)


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if path.name.startswith(".env"):
            continue
        if path.suffix not in INCLUDE_SUFFIXES and path.name not in INCLUDE_NAMES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path


def chunk_text(repo: str, rel_path: str, text: str, window: int = 60, overlap: int = 10) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    step = max(1, window - overlap)
    for start in range(0, len(lines), step):
        piece = lines[start:start + window]
        if sum(1 for line in piece if line.strip()) < 3:
            continue
        body = "\n".join(piece)
        if SECRET_PATTERN.search(body):
            continue
        first, last = start + 1, start + len(piece)
        chunks.append(Chunk(
            id=f"{repo}:{rel_path}:{first}-{last}",
            repo=repo, path=rel_path, start_line=first, end_line=last, text=body,
        ))
        if start + window >= len(lines):
            break
    return chunks


def chunk_repo(repo: str, root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_source_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if NO_INDEX_MARKER in text[:2000]:
            continue
        chunks.extend(chunk_text(repo, path.relative_to(root).as_posix(), text))
    return chunks


def chunk_profile(path: Path) -> list[Chunk]:
    """Split my profile (Markdown) into one chunk per `## ` section.

    These become *self-reported* evidence: shown as "Stated on CV", never as proof.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")] + [len(lines)]
    for begin, end in zip(starts, starts[1:]):
        title = lines[begin][3:].strip()
        body = "\n".join(lines[begin + 1:end]).strip()
        if not body or SECRET_PATTERN.search(body):
            continue
        chunks.append(Chunk(
            id=f"profile:{path.name}:{begin + 1}-{end}", repo="profile", path=path.name,
            start_line=begin + 1, end_line=end, text=body,
            source_type="self_reported", title=title,
        ))
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded: the file path (or CV section title) gives context the text alone lacks."""
    if chunk.source_type == "self_reported":
        return f"CV - {chunk.title}\n{chunk.text}"
    return f"{chunk.repo}/{chunk.path}\n{chunk.text}"
