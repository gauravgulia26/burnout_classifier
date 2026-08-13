"""Schemas for burnout-risk inference."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BurnoutInput(BaseModel):
    """Raw student attributes required to calculate a burnout-risk prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    Major_Category: str = Field(min_length=1, max_length=100)
    Year_of_Study: Literal["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
    Pre_Semester_GPA: float = Field(ge=0, le=4)
    Weekly_GenAI_Hours: float = Field(ge=0, le=168)
    Primary_Use_Case: str = Field(min_length=1, max_length=100)
    Prompt_Engineering_Skill: Literal["Beginner", "Intermediate", "Advanced"]
    Tool_Diversity: int = Field(ge=1)
    Paid_Subscription: bool
    Traditional_Study_Hours: float = Field(ge=0, le=168)
    Perceived_AI_Dependency: int = Field(ge=1, le=10)
    Institutional_Policy: Literal[
        "Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"
    ]
    Anxiety_Level_During_Exams: int = Field(ge=1, le=10)
    Post_Semester_GPA: float = Field(ge=0, le=4)
    Skill_Retention_Score: float = Field(ge=0, le=100)


class PredictionRequest(BaseModel):
    """One or more raw student records to score."""

    model_config = ConfigDict(extra="forbid")

    records: list[BurnoutInput] = Field(min_length=1, max_length=1_000)


class PredictionResult(BaseModel):
    """A single model result."""

    burnout_risk: Literal["Low", "Medium", "High"]


class PredictionResponse(BaseModel):
    """Predictions returned in the same order as the submitted records."""

    model_uri: str
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_uri: str

