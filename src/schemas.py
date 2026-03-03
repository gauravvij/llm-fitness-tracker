"""Pydantic schemas for structured Judge LLM output validation."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluationScore(BaseModel):
    """Schema for a single LLM response evaluation by the Judge."""

    accuracy: float = Field(default=5.0, ge=0.0, le=10.0, description="Accuracy score 0-10")
    hallucination: float = Field(default=5.0, ge=0.0, le=10.0, description="Anti-hallucination score 0-10")
    grounding: float = Field(default=5.0, ge=0.0, le=10.0, description="Grounding score 0-10")
    reasoning: float = Field(default=5.0, ge=0.0, le=10.0, description="Reasoning depth score 0-10")
    clarity: float = Field(default=5.0, ge=0.0, le=10.0, description="Clarity score 0-10")
    overall: float = Field(default=5.0, ge=0.0, le=10.0, description="Weighted overall score 0-10")
    reasoning_text: str = Field(
        default="No reasoning provided.",
        description="3-4 sentence critical justification",
    )

    @field_validator("accuracy", "hallucination", "grounding", "reasoning", "clarity", "overall", mode="before")
    @classmethod
    def coerce_float(cls, v):
        """Coerce numeric strings and clamp to valid range."""
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 5.0

    @field_validator("reasoning_text", mode="before")
    @classmethod
    def ensure_nonempty_reasoning(cls, v):
        """Ensure reasoning_text is never empty."""
        if not v or not str(v).strip():
            return "No detailed reasoning provided by the judge."
        return str(v).strip()

    @model_validator(mode="after")
    def recalculate_overall(self) -> "EvaluationScore":
        """Recalculate overall using the canonical weighted formula."""
        self.overall = round(
            self.accuracy * 0.35
            + self.hallucination * 0.20
            + self.grounding * 0.20
            + self.reasoning * 0.15
            + self.clarity * 0.10,
            2,
        )
        return self

    def to_dict(self) -> dict:
        """Return as plain dict with tool_calling alias for backward compatibility."""
        d = self.model_dump()
        d["tool_calling"] = d["reasoning"]  # backward compat alias
        return d


class RankingEntry(BaseModel):
    """Schema for a single model's ranking entry."""

    rank: int = Field(ge=1, le=10, description="Rank position (1 = best)")
    model_id: str = Field(min_length=1, description="OpenRouter model ID")
    overall_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Overall score 0-10")
    strengths: list[str] = Field(default_factory=list, description="List of model strengths")
    weaknesses: list[str] = Field(default_factory=list, description="List of model weaknesses")
    recommendation: str = Field(
        default="Recommended based on benchmark performance.",
        description="One sentence recommendation",
    )

    @field_validator("strengths", "weaknesses", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure strengths/weaknesses are always lists, never None."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle case where LLM returns a comma-separated string
            return [item.strip() for item in v.split(",") if item.strip()]
        return list(v)

    @field_validator("strengths", mode="after")
    @classmethod
    def ensure_nonempty_strengths(cls, v):
        """Ensure strengths has at least one entry."""
        if not v:
            return ["Strong overall benchmark performance"]
        return v

    @field_validator("weaknesses", mode="after")
    @classmethod
    def ensure_nonempty_weaknesses(cls, v):
        """Ensure weaknesses has at least one entry for honest reporting."""
        if not v:
            return ["No significant weaknesses identified in benchmark"]
        return v

    @field_validator("recommendation", mode="before")
    @classmethod
    def ensure_nonempty_recommendation(cls, v):
        """Ensure recommendation is never empty."""
        if not v or not str(v).strip():
            return "Recommended based on benchmark performance."
        return str(v).strip()

    @field_validator("overall_score", mode="before")
    @classmethod
    def coerce_score(cls, v):
        """Coerce numeric strings."""
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 5.0


class RankingResult(BaseModel):
    """Schema for the full ranking response from the Judge LLM."""

    ranking: list[RankingEntry] = Field(default_factory=list, description="Ordered list of top models")
    summary: str = Field(
        default="Models ranked by aggregated benchmark performance.",
        description="3-4 sentence overall analysis",
    )

    @field_validator("ranking", mode="before")
    @classmethod
    def ensure_ranking_list(cls, v):
        """Ensure ranking is always a list."""
        if v is None:
            return []
        return list(v)

    @field_validator("summary", mode="before")
    @classmethod
    def ensure_nonempty_summary(cls, v):
        """Ensure summary is never empty."""
        if not v or not str(v).strip():
            return "Models ranked by aggregated benchmark performance."
        return str(v).strip()

    def to_dict(self) -> dict:
        """Return as plain dict."""
        return {
            "ranking": [entry.model_dump() for entry in self.ranking],
            "summary": self.summary,
        }


class TestCase(BaseModel):
    """Schema for a generated test case."""

    id: int = Field(ge=1, description="Test case ID")
    category: str = Field(default="general", description="Test category")
    prompt: str = Field(min_length=10, description="The test prompt")
    evaluation_criteria: str = Field(
        default="Response should be accurate, relevant, and well-structured",
        description="What a correct response must contain",
    )
    expected_elements: list[str] = Field(
        default_factory=lambda: ["relevance", "accuracy", "clarity"],
        description="Key elements expected in a good response",
    )
    difficulty: str = Field(default="medium", description="easy|medium|hard")

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v):
        """Normalize category values."""
        valid = {"basic", "reasoning", "edge_case", "accuracy", "tool_calling", "general"}
        if not v or str(v).lower() not in valid:
            return "general"
        return str(v).lower()

    @field_validator("difficulty", mode="before")
    @classmethod
    def validate_difficulty(cls, v):
        """Normalize difficulty values."""
        valid = {"easy", "medium", "hard"}
        if not v or str(v).lower() not in valid:
            return "medium"
        return str(v).lower()

    @field_validator("expected_elements", mode="before")
    @classmethod
    def ensure_elements_list(cls, v):
        """Ensure expected_elements is always a list."""
        if v is None:
            return ["relevance", "accuracy", "clarity"]
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return list(v)
