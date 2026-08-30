"""
Веб-сервис классификации SMS-сообщений (spam / ham).

Модель загружается как целый sklearn Pipeline: признаки посчитаны
внутри него тем же кодом из features.py, что использовался при обучении.
Поэтому на вход эндпоинтов подаётся сырой текст, без предобработки.

Запуск:
    uvicorn src.app.main:app --reload
"""

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- пути и импорт общего модуля признаков -------------------------------
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
ROOT = SRC_DIR.parent
MODEL_DIR = ROOT / "models"

# Пайплайн сериализован со ссылкой на класс ManualFeatures из features.py,
# поэтому модуль обязан быть импортируемым до joblib.load
sys.path.insert(0, str(SRC_DIR))
from features import MANUAL_FEATURE_NAMES  # noqa: E402

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title="Классификатор SMS: spam / ham",
    description="Итоговый проект, вариант 3 — классические ML-модели",
    version="1.0.0",
)

# --- описание признаков для эндпоинта /features --------------------------
FEATURE_GROUPS = {
    "n_chars": ("statistical", "Длина текста в символах"),
    "n_words": ("statistical", "Количество слов"),
    "n_sentences": ("statistical", "Количество предложений"),
    "avg_word_len": ("statistical", "Средняя длина слова"),
    "n_unique_words": ("lexical", "Количество уникальных слов"),
    "lexical_diversity": ("lexical", "Отношение уникальных слов ко всем"),
    "n_stopwords": ("lexical", "Количество стоп-слов"),
    "stopword_ratio": ("lexical", "Доля стоп-слов"),
    "n_punct": ("syntactic", "Количество знаков препинания"),
    "n_upper": ("syntactic", "Количество заглавных букв"),
    "upper_ratio": ("syntactic", "Доля заглавных букв"),
    "n_digits": ("syntactic", "Количество цифр"),
    "digit_ratio": ("syntactic", "Доля цифр"),
    "n_exclamations": ("syntactic", "Количество восклицательных знаков"),
}

# --- загрузка модели -----------------------------------------------------
MODEL = None
META = {}
FEATURE_NAMES: list[str] = []
LOAD_ERROR = None


def load_model():
    """Загружает пайплайн и метаданные. Ошибку не гасим молча —
    показываем её в интерфейсе и в /model/info."""
    global MODEL, META, FEATURE_NAMES, LOAD_ERROR
    path = MODEL_DIR / "best_model.joblib"
    if not path.exists():
        LOAD_ERROR = (
            f"Файл модели не найден: {path}. "
            "Запустите обучение: python src/train.py"
        )
        return
    try:
        MODEL = joblib.load(path)
        FEATURE_NAMES = [
            str(n) for n in MODEL.named_steps["features"].get_feature_names_out()
        ]
        meta_path = MODEL_DIR / "model_meta.json"
        if meta_path.exists():
            META = json.loads(meta_path.read_text(encoding="utf-8"))
        LOAD_ERROR = None
    except Exception as exc:
        LOAD_ERROR = f"Не удалось загрузить модель: {exc}"


@app.on_event("startup")
def startup():
    load_model()


def require_model():
    if MODEL is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "Модель не загружена")


# --- вспомогательные функции --------------------------------------------
def manual_values(text: str) -> dict[str, float]:
    """Значения ручных признаков для одного текста — берём из самого пайплайна,
    чтобы не дублировать логику."""
    block = MODEL.named_steps["features"].transformer_list
    for name, tr in block:
        if name == "manual":
            raw = tr.named_steps["manual"].transform([text])[0]
            return dict(zip(MANUAL_FEATURE_NAMES, [float(v) for v in raw]))
    return {}


def per_sample_contributions(text: str) -> dict[str, float]:
    """Вклад каждого признака в конкретное предсказание.

    Для линейных моделей это coef * value, для LightGBM — встроенный
    pred_contrib (SHAP-значения). Если модель не поддерживает ни того,
    ни другого, возвращаем пустой словарь и интерфейс просто покажет
    значения признаков без вклада.
    """
    try:
        X = MODEL.named_steps["features"].transform([text])
        # LightGBM с pred_contrib не принимает разреженную матрицу,
        # а одна строка в плотном виде занимает несколько килобайт
        dense = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        clf = MODEL.named_steps["clf"]

        if hasattr(clf, "coef_"):
            contrib = clf.coef_[0] * dense.ravel()
        elif hasattr(clf, "booster_"):
            # последний столбец — базовое значение (bias), признаком не является
            contrib = np.asarray(clf.booster_.predict(dense, pred_contrib=True))[0][:-1]
        else:
            return {}

        return {
            name.replace("manual__", ""): float(c)
            for name, c in zip(FEATURE_NAMES, contrib)
            if name.startswith("manual__")
        }
    except Exception:
        return {}


def classify(text: str) -> dict:
    proba = float(MODEL.predict_proba([text])[0, 1])
    is_spam = proba >= 0.5
    values = manual_values(text)
    contrib = per_sample_contributions(text)

    features = [
        {"name": n, "value": values.get(n, 0.0), "contribution": contrib.get(n)}
        for n in MANUAL_FEATURE_NAMES
    ]
    # Самые повлиявшие признаки — вперёд
    features.sort(key=lambda f: abs(f["contribution"] or 0), reverse=True)

    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "confidence": proba if is_spam else 1 - proba,
        "proba_spam": proba,
        "text_preview": text[:200],
        "features": features,
    }


# --- эндпоинты -----------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Главная страница")
def index(request: Request):
    context = {
        "model_name": META.get("model_name", "не загружена"),
        "metrics": META.get("metrics_test", {}),
        "error": LOAD_ERROR,
    }
    # В свежих версиях Starlette сигнатура — (request, name, context),
    # в старых — (name, context) с request внутри контекста
    try:
        return templates.TemplateResponse(request, "index.html", context)
    except TypeError:
        return templates.TemplateResponse("index.html", {"request": request, **context})


@app.post("/predict", summary="Классификация текста")
def predict(payload: dict):
    require_model()
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="Поле text пустое")
    if len(text) > 10_000:
        raise HTTPException(status_code=422, detail="Текст длиннее 10 000 символов")
    return classify(text)


@app.post("/predict/file", summary="Классификация из файла")
async def predict_file(file: UploadFile = File(...)):
    require_model()

    if not file.filename.lower().endswith((".txt", ".csv")):
        raise HTTPException(status_code=422, detail="Поддерживаются только .txt и .csv")

    raw = await file.read()
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Файл больше 2 МБ")

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    lines = [ln.strip() for ln in io.StringIO(content) if ln.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="В файле нет строк")
    lines = lines[:500]  # ограничение на один запрос

    probas = MODEL.predict_proba(lines)[:, 1]
    results = [
        {
            "line": i + 1,
            "text_preview": t[:120],
            "label": "spam" if p >= 0.5 else "ham",
            "proba_spam": round(float(p), 4),
        }
        for i, (t, p) in enumerate(zip(lines, probas))
    ]
    spam_count = sum(r["label"] == "spam" for r in results)

    return {
        "filename": file.filename,
        "total": len(results),
        "spam_count": spam_count,
        "ham_count": len(results) - spam_count,
        "results": results,
    }


@app.get("/features", summary="Список используемых признаков")
def features():
    require_model()
    manual = [
        {"name": n, "group": FEATURE_GROUPS[n][0], "description": FEATURE_GROUPS[n][1]}
        for n in MANUAL_FEATURE_NAMES
    ]
    return {
        "total": len(FEATURE_NAMES),
        "manual_count": len(MANUAL_FEATURE_NAMES),
        "vector_count": len(FEATURE_NAMES) - len(MANUAL_FEATURE_NAMES),
        "manual": manual,
    }


@app.get("/feature/importance", summary="Важность признаков")
def feature_importance(top_n: int = 25):
    require_model()
    clf = MODEL.named_steps["clf"]

    if hasattr(clf, "coef_"):
        values, kind = clf.coef_[0], "coefficients"
    elif hasattr(clf, "feature_importances_"):
        values, kind = clf.feature_importances_, "feature_importances"
    else:
        raise HTTPException(status_code=501, detail="Модель не сообщает важность признаков")

    weights = np.abs(values)
    is_manual = np.array([n.startswith("manual__") for n in FEATURE_NAMES])
    manual_share = float(weights[is_manual].sum() / weights.sum()) if weights.sum() else 0.0

    order = np.argsort(-weights)[: max(1, min(top_n, len(FEATURE_NAMES)))]
    top = [
        {
            "name": FEATURE_NAMES[i].replace("manual__", "").replace("tfidf__", ""),
            "importance": round(float(values[i]), 6),
            "group": "manual" if is_manual[i] else "tfidf",
        }
        for i in order
    ]

    return {
        "model_name": META.get("model_name", type(clf).__name__),
        "kind": kind,
        "manual_share": round(manual_share, 4),
        "top": top,
    }


@app.get("/model/info", summary="Информация о модели")
def model_info():
    require_model()
    return {
        "model_name": META.get("model_name", "unknown"),
        "feature_set": META.get("feature_set", "combined"),
        "n_features": len(FEATURE_NAMES),
        "metrics_test": META.get("metrics_test", {}),
        "best_params": META.get("best_params", {}),
    }
