"""Train and serve a calibrated query-ad click-through-rate model.

The default interactions are synthetic because the supplied catalog contains no
impression/click logs. Never interpret these predictions as real market CTRs.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from .features import build_numeric_features, normalize_query

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"


def product_text(row):
    return f"{row.get('product', '')} {row.get('category', '')} {row.get('description', '')}"



class CalibratedCTR:
    def __init__(self, vectorizer, classifier, calibrator):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.calibrator = calibrator

    def predict(self, query, ads):
        queries = [query] * len(ads)
        docs = [product_text(row) for _, row in ads.iterrows()]
        doc_matrix = self.vectorizer.transform(docs)
        q_matrix = self.vectorizer.transform([normalize_query(query)] * len(ads))
        sims = np.asarray(q_matrix.multiply(doc_matrix).sum(axis=1)).ravel()
        numeric = np.vstack([build_numeric_features(query, row, sim) for (_, row), sim in zip(ads.iterrows(), sims)])
        # Pairwise text signal is represented by cosine similarity plus interpretable numeric features.
        pair_text = q_matrix.multiply(doc_matrix)
        X = hstack([pair_text, csr_matrix(numeric)], format="csr")
        raw = self.classifier.predict_proba(X)[:, 1]
        return np.clip(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], 0.001, 0.999), sims


def train_model(interactions_path=None, model_dir=MODEL_DIR):
    path = Path(interactions_path) if interactions_path else DATA / "interactions_synthetic.csv"
    df = pd.read_csv(path)
    ads = pd.read_csv(DATA / "advertisers.csv").set_index("advertiser_id")
    joined = df.join(ads[["product", "category", "description", "price_inr"]], on="advertiser_id", rsuffix="_ad")
    joined = joined.dropna(subset=["query", "click", "product"])
    texts = [product_text(row) for _, row in joined.iterrows()]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, stop_words="english")
    vectorizer.fit(texts + [normalize_query(q) for q in joined["query"].astype(str).tolist()])
    docs = vectorizer.transform(texts)
    qs = vectorizer.transform([normalize_query(q) for q in joined["query"].astype(str).tolist()])
    sims = np.asarray(qs.multiply(docs).sum(axis=1)).ravel()
    numeric = np.vstack([
        build_numeric_features(q, row, sim)
        for q, (_, row), sim in zip(joined["query"].astype(str), joined.iterrows(), sims)
    ])
    # Pairwise query/ad token interactions plus numeric context; bid is deliberately excluded.
    pair_text = qs.multiply(docs)
    X = hstack([pair_text, csr_matrix(numeric)], format="csr")
    y = joined["click"].astype(int).to_numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43, stratify=y_temp)
    classifier = LogisticRegression(max_iter=1000, class_weight=None)
    classifier.fit(X_train, y_train)
    cal_raw = classifier.predict_proba(X_cal)[:, 1]
    calibrator = LogisticRegression(max_iter=1000)  # Platt-style sigmoid calibration
    calibrator.fit(cal_raw.reshape(-1, 1), y_cal)
    test_raw = classifier.predict_proba(X_test)[:, 1]
    test_cal = np.clip(calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1], 0.001, 0.999)
    metrics = {
        "training_data": "synthetic" if "synthetic" in path.name else "provided interaction log",
        "rows": int(len(df)),
        "train_rows": int(len(y_train)), "calibration_rows": int(len(y_cal)), "test_rows": int(len(y_test)),
        "test_auc": float(roc_auc_score(y_test, test_cal)),
        "test_log_loss": float(log_loss(y_test, test_cal)),
        "test_brier_score": float(brier_score_loss(y_test, test_cal)),
        "mean_test_predicted_ctr": float(np.mean(test_cal)),
        "observed_test_ctr": float(np.mean(y_test)),
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "calibrator": calibrator}, model_dir / "ctr_model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(model_dir=MODEL_DIR):
    obj = joblib.load(Path(model_dir) / "ctr_model.joblib")
    return CalibratedCTR(obj["vectorizer"], obj["classifier"], obj["calibrator"])
