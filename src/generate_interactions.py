"""Create a reproducible synthetic impression/click log for the demo only."""
from pathlib import Path
import numpy as np
import pandas as pd
from .features import build_numeric_features, detect_category

ROOT = Path(__file__).resolve().parents[1]

QUERIES = [
    "laptop for gaming", "laptop for programming", "laptop for students", "laptop for machine learning",
    "laptop under ₹80000", "laptop under ₹1 lakh with 16GB RAM", "lightweight laptop with good battery",
    "smartphone for photography", "smartphone with good battery", "phone under ₹30000",
    "tablet for note taking", "tablet for online classes", "headphones for online meetings",
    "noise cancelling headphones", "monitor for coding", "high refresh rate gaming monitor",
    "camera for travel photography", "camera for video creation", "mechanical keyboard for gaming",
    "wireless mouse for productivity", "printer for home office", "portable external ssd for backups",
    "fast storage for laptop", "power bank for travel", "wifi router for gaming",
    "smartwatch with long battery", "usb hub for laptop", "accessories for workstation",
    "budget tablet for students", "professional laptop for development", "gaming accessories",
    "computer components for upgrade", "storage device for large files", "phone for everyday use",
    "monitor for office work", "camera for content creation", "headphones for music",
    "laptop with 16GB RAM", "laptop with 512GB SSD", "smartphone under ₹50000",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(output_path=None, rows_per_query=600, seed=2026):
    rng = np.random.default_rng(seed)
    catalog = pd.read_csv(ROOT / "data" / "advertisers.csv")
    records = []
    for query in QUERIES:
        # Mix relevant category candidates with random catalog distractors.
        category = detect_category(query)
        if category:
            relevant = catalog[catalog.category == category]
        else:
            relevant = catalog
        n_rel = min(int(rows_per_query * 0.65), len(relevant))
        n_rand = rows_per_query - n_rel
        picked_rel = relevant.sample(n=n_rel, replace=n_rel > len(relevant), random_state=int(rng.integers(0, 2**31-1)))
        picked_rand = catalog.sample(n=n_rand, replace=n_rand > len(catalog), random_state=int(rng.integers(0, 2**31-1)))
        candidates = pd.concat([picked_rel, picked_rand]).drop_duplicates("advertiser_id")
        budget = None
        import re
        from .features import parse_budget
        budget = parse_budget(query)
        for _, ad in candidates.iterrows():
            f = build_numeric_features(query, ad, 0.0)
            # Synthetic click propensity: category/query match, textual overlap and budget fit.
            # This is a simulation rule, not empirical user behaviour.
            logit = -3.15 + 2.5*f[1] + 3.1*f[2] + 0.85*f[3] - 0.30*f[4]
            p = float(np.clip(sigmoid(logit), 0.005, 0.75))
            records.append({"query": query, "advertiser_id": ad.advertiser_id,
                            "click_probability_simulated": p,
                            "click": int(rng.random() < p)})
    df = pd.DataFrame(records)
    out = Path(output_path) if output_path else ROOT / "data" / "interactions_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df

if __name__ == "__main__":
    result = generate()
    print(f"Wrote {len(result):,} synthetic impressions; simulated CTR={result.click.mean():.3%}")
