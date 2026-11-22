"""Advertisement retrieval using sentence embeddings, with a TF-IDF fallback."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "advertisers.csv"

class AdRetriever:
    def __init__(self, data_path=DATA_PATH, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.ads = pd.read_csv(data_path).fillna("")
        self.model = None
        self.embeddings = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            texts = (self.ads["product"] + ". " + self.ads["description"]).tolist()
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        except Exception as exc:
            # Offline-friendly fallback. It is lexical retrieval, not semantic embedding retrieval.
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            texts = (self.ads["product"] + ". " + self.ads["description"]).tolist()
            self.embeddings = self.vectorizer.fit_transform(texts)
            self.fallback_reason = str(exc)

    def retrieve(self, query, top_k=5):
        if self.model is not None:
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.embeddings @ q
        else:
            q = self.vectorizer.transform([query])
            scores = (self.embeddings @ q.T).toarray().ravel()
            # Normalize lexical scores to [0,1] for a convenient relevance scale.
            max_score = scores.max() if len(scores) else 0
            scores = scores / max_score if max_score > 0 else scores
        result = self.ads.copy()
        result["relevance"] = scores
        result = result.sort_values("relevance", ascending=False).head(top_k)
        # Keep the auction relevance strictly positive, as required by this demo's score rule.
        result["relevance"] = result["relevance"].clip(lower=0.01, upper=1.0)
        return result.reset_index(drop=True)
