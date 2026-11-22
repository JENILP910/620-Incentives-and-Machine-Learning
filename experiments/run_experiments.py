"""Monte Carlo comparison of allocation rules on the sample ad catalogue."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.auction import run_auction
from src.baselines import select_baseline
from src.retriever import AdRetriever

def main(trials=1000):
    retriever = AdRetriever()
    ads = retriever.retrieve("laptop for machine learning programming with 16GB RAM", top_k=5)
    rng = np.random.default_rng(7)
    methods = ["highest_bid", "highest_relevance", "bid_times_relevance", "randomized_auction"]
    records = []
    for method in methods:
        wins, revenue, welfare, relevance = 0, 0.0, 0.0, 0.0
        for _ in range(trials):
            if method == "randomized_auction":
                result = run_auction(ads, rng=rng)
                winner = result["winner"]
                payment = result["payment"]
            else:
                key = {"highest_bid":"bid", "highest_relevance":"relevance",
                       "bid_times_relevance":"bid_x_relevance"}[method]
                winner = select_baseline(ads, key)
                payment = 0.0  # Baseline payments are not modelled in this simple comparison.
            wins += 1
            revenue += payment
            welfare += float(winner["relevance"]) * float(winner["bid_inr_per_click"])
            relevance += float(winner["relevance"])
        records.append({"method":method, "trials":trials,
                        "avg_payment_inr_per_trial":revenue/trials,
                        "avg_relevance":relevance/trials,
                        "avg_relevance_x_bid_proxy":welfare/trials})
    out = pd.DataFrame(records)
    path = ROOT / "experiments" / "results.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
