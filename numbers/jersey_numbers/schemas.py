from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NumbersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OCRPrediction(NumbersModel):
    track_id: int
    frame_id: int
    predicted_number: str
    confidence: float
    model_name: str


class MatchedNumber(NumbersModel):
    track_id: int
    frame_id: int
    number_bbox_xyxy: list[float]
    ios_score: float
    predicted_number: str | None = None
    confidence: float | None = None

    @field_validator("number_bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("number_bbox_xyxy must contain exactly four values")
        return [float(v) for v in value]


class ConfirmedPlayer(NumbersModel):
    track_id: int
    jersey_number: str
    team: str | None = None
    player_name: str | None = None


class NumbersOutput(NumbersModel):
    video: str
    ocr_model: str
    ios_threshold: float
    confirmed_players: list[ConfirmedPlayer] = Field(default_factory=list)
    total_frames_processed: int = 0
