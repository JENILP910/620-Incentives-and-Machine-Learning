"""Constraint-aware product retrieval followed by calibrated CTR scoring."""
from pathlib import Path
import pandas as pd
from .features import parse_budget, detect_category, explicit_constraints, parse_ad_specs
from .ctr_model import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "advertisers.csv"

class AdRetriever:
    def __init__(self, data_path=DATA_PATH, ctr_model=None):
        self.ads = pd.read_csv(data_path).fillna("")
        self.ctr_model = ctr_model or load_model()

    def filter_candidates(self, query):
        rows = self.ads.copy()
        budget = parse_budget(query)
        category = detect_category(query)
        constraints = explicit_constraints(query)
        if category:
            rows = rows[rows.category.str.lower() == category]
        if budget is not None:
            rows = rows[pd.to_numeric(rows.price_inr, errors="coerce") <= budget]
        if constraints:
            for key, required in constraints.items():
                if rows.empty:
                    break
                specs = rows.apply(parse_ad_specs, axis=1, result_type="expand")
                rows = rows.loc[specs[key].notna() & (specs[key] >= required)]
        return rows.reset_index(drop=True), {"budget_inr": budget, "category": category, "specifications": constraints}

    def retrieve(self, query, top_k=10):
        candidates, filters = self.filter_candidates(query)
        if candidates.empty:
            return candidates.assign(relevance=[], ctr_estimate=[], semantic_similarity=[]), filters
        ctr, similarity = self.ctr_model.predict(query, candidates)
        candidates["semantic_similarity"] = similarity
        candidates["ctr_estimate"] = ctr
        # The calibrated CTR is the paper-aligned relevance proxy q_i.
        candidates["relevance"] = candidates["ctr_estimate"].clip(lower=0.001)
        candidates = candidates.sort_values("relevance", ascending=False).head(top_k)
        return candidates.reset_index(drop=True), filters
