"""Shared Pydantic models used across all patterns."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Documents ─────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    URL_EXTRACT = "url_extract"


class Document(BaseModel):
    """A retrieved source document."""
    id: str = ""
    title: str = ""
    content: str = ""
    url: str = ""
    source_type: SourceType = SourceType.WEB
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def short_id(self) -> str:
        return self.id[:12] if self.id else self.url[:40]


# ── Citations ─────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A citation linking a claim to a source."""
    claim: str = ""
    source_id: str = ""
    source_title: str = ""
    source_url: str = ""
    relevance_score: float = 0.0


# ── Research artifacts ────────────────────────────────────────────────────────

class SubQuery(BaseModel):
    """A decomposed sub-query."""
    query: str
    intent: str = ""
    priority: int = 1


class Section(BaseModel):
    """A section of the final report."""
    title: str
    content: str
    citations: List[Citation] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final output of any pattern."""
    query: str
    title: str = ""
    abstract: str = ""
    sections: List[Section] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pattern_name: str = ""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def full_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"# {self.title}\n")
        if self.abstract:
            parts.append(f"## Abstract\n{self.abstract}\n")
        for s in self.sections:
            parts.append(f"## {s.title}\n{s.content}\n")
        if self.citations:
            parts.append("## References\n")
            for i, c in enumerate(self.citations, 1):
                parts.append(f"[{i}] {c.source_title} — {c.source_url}")
        return "\n".join(parts)


# ── LLM usage tracking ───────────────────────────────────────────────────────

class LLMUsage(BaseModel):
    """Single LLM call usage record."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    call_type: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Pattern-specific types ────────────────────────────────────────────────────

class Perspective(BaseModel):
    """A research perspective (for STORM pattern)."""
    name: str
    description: str
    focus_areas: List[str] = Field(default_factory=list)


class TopicCluster(BaseModel):
    """A cluster of related information (for MERIDIAN pattern)."""
    topic: str
    summary: str = ""
    source_ids: List[str] = Field(default_factory=list)
    importance: float = 0.0


class WDAllocation(BaseModel):
    """Width-Depth budget allocation (for Hierarchical W&D)."""
    step: int
    width_budget: float
    depth_budget: float
    width_workers: int
    depth_iterations: int


# ── Process trajectory tracking (E8) ──────────────────────────────────────────

class ToolCall(BaseModel):
    """One step of a pattern's tool/reasoning trace.

    Patterns log a sequence of these via StateManager so the trajectory rubric
    (E8) can score retrieval quality, reasoning coherence, iterative refinement,
    and tool efficiency separately from final-report outcome scoring.
    """
    step_idx: int
    tool: str  # "search" | "academic_search" | "read" | "extract" | "generate" | "reflect" | "decompose" | ...
    input_args: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = Field(
        default="",
        description="Human-readable short summary of the tool's output (≤500 chars).",
    )
    n_results: Optional[int] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessTrace(BaseModel):
    """A pattern's full tool-call sequence for one query.

    Saved alongside ResearchReport via StateManager.save("trace", trace.model_dump()).
    The trajectory rubric (deep_research/evaluation/trajectory_rubric.py) consumes
    this to compute process-quality scores complementary to outcome scoring.
    """
    pattern_name: str
    query: str
    query_id: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    n_search_queries: int = 0
    n_unique_urls_visited: int = 0
    n_iterations: int = 0  # for iterative patterns (P1 RAG, P5 W&D, P6 ReAct, P11 ReAct)
    final_report_word_count: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    def append(self, **kwargs) -> ToolCall:
        """Convenience: build + append a ToolCall at the next step_idx."""
        kwargs.setdefault("step_idx", len(self.tool_calls))
        call = ToolCall(**kwargs)
        self.tool_calls.append(call)
        if call.tool in ("search", "academic_search"):
            self.n_search_queries += 1
        return call
