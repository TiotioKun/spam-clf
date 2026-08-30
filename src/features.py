"""
Модуль извлечения признаков для классификации текстов.

ВАЖНО: этот модуль импортируется и исследовательским ноутбуком,
и веб-сервисом. Не дублируйте логику признаков нигде больше —
иначе обучение и инференс разойдутся.
"""

import re
import string

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# Английские стоп-слова. Для русскоязычного датасета замените
# на nltk.corpus.stopwords.words('russian') или список из pymorphy.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now",
}

_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD_SPLIT = re.compile(r"\b\w+\b")
_PUNCT = set(string.punctuation)

# Имена признаков в том же порядке, в каком их возвращает transform().
# Нужны для эндпоинта /features и для графиков важности.
MANUAL_FEATURE_NAMES = [
    "n_chars",           # длина текста в символах
    "n_words",           # количество слов
    "n_sentences",       # количество предложений
    "avg_word_len",      # средняя длина слова
    "n_unique_words",    # количество уникальных слов
    "lexical_diversity",  # уникальные / все слова
    "n_stopwords",       # количество стоп-слов
    "stopword_ratio",    # доля стоп-слов
    "n_punct",           # количество знаков препинания
    "n_upper",           # количество заглавных букв
    "upper_ratio",       # доля заглавных букв
    "n_digits",          # количество цифр
    "digit_ratio",       # доля цифр
    "n_exclamations",    # количество восклицательных знаков
]


def extract_manual_features(text: str) -> list[float]:
    """Считает 14 «ручных» признаков для одного текста."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    n_chars = len(text)
    words = _WORD_SPLIT.findall(text.lower())
    n_words = len(words)

    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)

    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    unique_words = set(words)
    n_unique = len(unique_words)
    lexical_diversity = n_unique / n_words if n_words else 0.0

    n_stop = sum(1 for w in words if w in STOP_WORDS)
    stop_ratio = n_stop / n_words if n_words else 0.0

    n_punct = sum(1 for ch in text if ch in _PUNCT)
    n_upper = sum(1 for ch in text if ch.isupper())
    n_digits = sum(1 for ch in text if ch.isdigit())
    denom = n_chars if n_chars else 1

    return [
        float(n_chars),
        float(n_words),
        float(n_sentences),
        avg_word_len,
        float(n_unique),
        lexical_diversity,
        float(n_stop),
        stop_ratio,
        float(n_punct),
        float(n_upper),
        n_upper / denom,
        float(n_digits),
        n_digits / denom,
        float(text.count("!")),
    ]


class ManualFeatures(BaseEstimator, TransformerMixin):
    """Sklearn-трансформер над «ручными» признаками.

    Нужен, чтобы признаки можно было положить внутрь Pipeline
    и сохранить вместе с моделью одним joblib-файлом.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return np.asarray([extract_manual_features(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MANUAL_FEATURE_NAMES, dtype=object)

    def to_frame(self, X) -> pd.DataFrame:
        """Удобная обёртка для EDA — возвращает DataFrame вместо массива."""
        return pd.DataFrame(self.transform(X), columns=MANUAL_FEATURE_NAMES)


def build_feature_union(
    kind: str = "combined",
    max_features: int = 3000,
    ngram_range: tuple = (1, 2),
) -> FeatureUnion | Pipeline:
    """Собирает блок признаков.

    kind:
        'manual'   — только ручные признаки (14 штук)
        'tfidf'    — только TF-IDF
        'count'    — только CountVectorizer
        'combined' — TF-IDF + ручные (основной вариант)

    Три первых варианта нужны для обязательного пункта задания
    «сравнить качество моделей с разными наборами признаков».
    """
    manual_block = Pipeline([
        ("manual", ManualFeatures()),
        ("scale", StandardScaler()),
    ])
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )
    count = CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)

    if kind == "manual":
        return manual_block
    if kind == "tfidf":
        return tfidf
    if kind == "count":
        return count
    if kind == "combined":
        return FeatureUnion([("tfidf", tfidf), ("manual", manual_block)])

    raise ValueError(f"Неизвестный набор признаков: {kind}")


def get_all_feature_names(fitted_union) -> list[str]:
    """Имена всех признаков обученного FeatureUnion — для графика важности."""
    return [str(n) for n in fitted_union.get_feature_names_out()]
