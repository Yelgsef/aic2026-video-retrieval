from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


QueryType = Literal["KIS", "QA", "TRAKE"]


class QueryRequest(BaseModel):
    type: QueryType
    query: str = Field(min_length=1, max_length=4000)
    query_transformer: bool = False
    top_k: int = Field(default=100, ge=1, le=1000)
    candidate_k: int = Field(default=300, ge=1, le=2000)
    rerank_k: int = Field(default=30, ge=0, le=500)
    openclip: bool = True
    reranker_enabled: bool = True


class QueryResponse(BaseModel):
    type: QueryType
    data: dict


class QaImage(BaseModel):
    data_url: str = Field(min_length=1)
    label: str | None = None


class QaAnswerRequest(BaseModel):
    provider: Literal["ai_studio", "9router"] = "ai_studio"
    api_key: str = ""
    model: str = Field(min_length=1, max_length=160)
    system_prompt: str = ""
    question: str = Field(min_length=1, max_length=4000)
    images: list[QaImage] = Field(min_length=1, max_length=16)


class QaModelsRequest(BaseModel):
    provider: Literal["ai_studio", "9router"] = "ai_studio"
    api_key: str = ""
