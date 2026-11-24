"""Compare ranking by bid, semantic similarity and calibrated CTR×bid."""
import numpy as np
import pandas as pd
from src.retriever import AdRetriever
from src.baselines import select_baseline
from src.auction import run_auction

QUERIES = ["gaming laptop", "laptop for machine learning", "smartphone for photography",
           "tablet for students", "monitor for coding", "headphones for meetings",
           "portable external ssd", "camera for travel"]

if __name__ == "__main__":
    retriever = AdRetriever()
    rows = []
    for i, query in enumerate(QUERIES):
        candidates, filters = retriever.retrieve(query, top_k=30)
        if candidates.empty:
            continue
        winner = run_auction(candidates, rng=np.random.default_rng(100+i))["winner"]
        bid_winner = select_baseline(candidates, "bid")
        semantic_winner = candidates.loc[candidates.semantic_similarity.idxmax()].to_dict()
        ctr_bid_winner = candidates.assign(score=candidates.ctr_estimate*candidates.bid_inr_per_click).sort_values("score", ascending=False).iloc[0].to_dict()
        rows.append({"query": query, "auction_winner": winner["product"],
                     "bid_only": bid_winner["product"], "semantic_only": semantic_winner["product"],
                     "calibrated_ctr_x_bid": ctr_bid_winner["product"],
                     "winner_ctr_estimate": winner["ctr_estimate"]})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("evaluation_results.csv", index=False)
    print("\nSaved evaluation_results.csv")
