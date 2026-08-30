"""Pydantic-схемы: валидация входа и описание формата ответа.

Схемы нужны не только для проверки данных — FastAPI строит по ним
автодокументацию на /docs, которую удобно показать на защите.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Текст сообщения для классификации",
        examples=["WIN a FREE iPhone! Call 09012345678 NOW!!!"],
    )


class FeatureValue(BaseModel):
    name: str
    value: float
    contribution: float | None = Field(
        None, description="Вклад признака в это конкретное предсказание"
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description="Предсказанный класс: spam или ham")
    is_spam: bool
    confidence: float = Field(..., ge=0, le=1, description="Уверенность в предсказанном классе")
    proba_spam: float = Field(..., ge=0, le=1)
    text_preview: str
    features: list[FeatureValue]


class BatchItem(BaseModel):
    line: int
    text_preview: str
    label: str
    proba_spam: float


class BatchResponse(BaseModel):
    filename: str
    total: int
    spam_count: int
    ham_count: int
    results: list[BatchItem]


class FeatureInfo(BaseModel):
    name: str
    group: str = Field(..., description="Группа: statistical / lexical / syntactic / vector")
    description: str


class FeaturesResponse(BaseModel):
    total: int
    manual_count: int
    vector_count: int
    manual: list[FeatureInfo]


class ImportanceItem(BaseModel):
    name: str
    importance: float
    group: str


class ImportanceResponse(BaseModel):
    model_name: str
    kind: str = Field(..., description="coefficients или feature_importances")
    manual_share: float = Field(..., description="Доля суммарного веса у ручных признаков")
    top: list[ImportanceItem]


class ModelInfo(BaseModel):
    model_name: str
    feature_set: str
    n_features: int
    metrics_test: dict
    best_params: dict
