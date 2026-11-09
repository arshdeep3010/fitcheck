"""Domain models shared by every layer (graph, API, evals)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """One thing the job description asks for."""

    id: str
    text: str
    must_have: bool = True
    # False for things code can't prove: years of experience, degrees, location, availability.
    checkable: bool = True


SourceType = Literal["code", "self_reported"]


class Chunk(BaseModel):
    """A slice of evidence. Line numbers are 1-based, inclusive.

    source_type "code" = a file from one of my repos (proof).
    source_type "self_reported" = a section of my profile/CV file (a claim, not proof).
    """

    id: str
    repo: str
    path: str
    start_line: int
    end_line: int
    text: str
    source_type: SourceType = "code"
    title: str = ""  # profile section heading, for self-reported chunks
    ref: str = ""  # commit SHA for repos synced from GitHub -> permalink citations
    url_base: str = ""  # e.g. https://github.com/me/repo, for repos synced from GitHub

    @property
    def source(self) -> str:
        if self.source_type == "self_reported":
            return f"CV: {self.title}" if self.title else "CV"
        return f"{self.repo}/{self.path}:L{self.start_line}-L{self.end_line}"


class Evidence(BaseModel):
    """A chunk retrieved for a requirement, with its cosine similarity."""

    chunk: Chunk
    similarity: float
    # Which search found it (hybrid retrieval): "vector", "keyword" or "both".
    via: Literal["vector", "keyword", "both"] = "vector"


Status = Literal["met", "partial", "not_met"]


class RequirementScore(BaseModel):
    """What the fit scorer (a decision model) says about one requirement.

    This is an opinion, not a verdict: the grounding step decides what is reported.
    """

    requirement_id: str
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_id: str | None = None  # chunk id the scorer picked as proof, None if none fits


class Verdict(str, Enum):
    MET = "met"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    UNVERIFIED = "unverified"  # the model may be right, but we can't prove it, so we don't claim it
    NOT_CHECKABLE = "not_checkable"  # years, degrees, location: not provable from code, not scored
    SELF_REPORTED = "self_reported"  # not proven in code, but my CV states it: shown as "Stated on CV"


CitationKind = Literal["github", "code", "cv"]


class Citation(BaseModel):
    """Where a verdict's evidence came from.

    kind "github" = code from a repo synced from GitHub (proof, permalink to the commit)
    kind "code"   = code from a local folder that isn't on GitHub yet (proof, but no public link)
    kind "cv"     = a section of the CV (a claim, never proof)
    """

    source: str
    url: str | None = None
    snippet: str
    kind: CitationKind = "code"
    repo: str = ""
    commit: str | None = None  # short SHA the GitHub link points at


class RequirementResult(BaseModel):
    requirement: Requirement
    verdict: Verdict
    confidence: float
    citation: Citation | None = None
    note: str = ""


class FitReport(BaseModel):
    candidate: str
    score: int = Field(ge=0, le=100)  # proven in code only
    score_with_cv: int = Field(ge=0, le=100, default=0)  # also counting "Stated on CV" rows
    summary: str
    results: list[RequirementResult]
    warnings: list[str] = []
    timings_ms: dict[str, int] = {}
    models: dict[str, str] = {}
