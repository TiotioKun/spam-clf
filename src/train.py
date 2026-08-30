"""
Обучение модели из командной строки.

Нужен для воспроизводимости: проверяющий может получить модель,
не открывая ноутбук. Логика повторяет ноутбук 02_models.ipynb,
без графиков и промежуточного анализа.

Запуск:
    python src/train.py
    python src/train.py --data data/spam.csv --n-iter 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import MANUAL_FEATURE_NAMES, build_feature_union  # noqa: E402

RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Читает датасет в двух форматах: TSV из UCI и CSV с Kaggle."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="latin-1").iloc[:, :2]
        df.columns = ["label", "text"]
    else:
        df = pd.read_csv(path, sep="\t", names=["label", "text"])

    before = len(df)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text", "label"])
    df["target"] = (df["label"] == "spam").astype(int)
    print(f"Загружено {before} строк, после очистки {len(df)}, "
          f"доля спама {df['target'].mean():.3f}")
    return df.reset_index(drop=True)


def metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }


def main():
    ap = argparse.ArgumentParser(description="Обучение классификатора spam/ham")
    ap.add_argument("--data", default=None, help="Путь к датасету")
    ap.add_argument("--n-iter", type=int, default=20, help="Итераций RandomizedSearchCV")
    ap.add_argument("--max-features", type=int, default=3000, help="Размер словаря TF-IDF")
    args = ap.parse_args()

    data_dir, model_dir = ROOT / "data", ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        data_path = Path(args.data)
    else:
        found = [p for p in (data_dir / "spam.csv", data_dir / "SMSSpamCollection")
                 if p.exists()]
        if not found:
            sys.exit(f"Датасет не найден в {data_dir}. Укажите путь через --data. "
                     "См. инструкцию в README.")
        data_path = found[0]

    df = load_data(data_path)
    X, y = df["text"].values, df["target"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")

    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        grid = {
            "clf__n_estimators": [100, 300, 500],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [-1, 5, 10],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__class_weight": ["balanced", None],
            "features__tfidf__max_features": [1000, 3000],
        }
        name = "LightGBM"
    except ImportError:
        print("LightGBM не установлен, откатываемся на RandomForest")
        clf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
        grid = {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 20, 50],
            "clf__min_samples_split": [2, 5, 10],
            "clf__class_weight": ["balanced", "balanced_subsample"],
            "features__tfidf__max_features": [1000, 3000],
        }
        name = "RandomForest"

    pipe = Pipeline([
        ("features", build_feature_union("combined", max_features=args.max_features)),
        ("clf", clf),
    ])

    print(f"\nТюнинг {name}: {args.n_iter} итераций, 5-fold CV...")
    t0 = time.time()
    search = RandomizedSearchCV(pipe, grid, n_iter=args.n_iter, scoring="f1",
                                cv=5, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Готово за {time.time() - t0:.0f} с. CV F1 = {search.best_score_:.4f}")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")

    best = search.best_estimator_
    val_m, test_m = metrics(best, X_val, y_val), metrics(best, X_test, y_test)
    print("\n" + pd.DataFrame({"Валидация": val_m, "Тест": test_m}).T.round(4).to_string())

    path = model_dir / "best_model.joblib"
    joblib.dump(best, path, compress=3)

    feat_names = [str(n) for n in best.named_steps["features"].get_feature_names_out()]
    meta = {
        "model_name": name,
        "feature_set": "combined",
        "n_features": len(feat_names),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "metrics_test": {k: round(float(v), 4) for k, v in test_m.items()},
        "manual_features": MANUAL_FEATURE_NAMES,
        "sklearn_version": sklearn.__version__,
    }
    (model_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nСохранено: {path} ({path.stat().st_size / 1024:.0f} КБ)")
    print("Запуск сервиса: uvicorn src.app.main:app --reload")


if __name__ == "__main__":
    main()
