"""Simple deterministic baselines for auction comparison."""
def select_baseline(ads, method="bid"):
    rows = ads.copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError("No candidate ads.")
    if method == "bid":
        idx = rows["bid_inr_per_click"].astype(float).idxmax()
    elif method == "relevance":
        idx = rows["relevance"].astype(float).idxmax()
    elif method == "bid_x_relevance":
        idx = (rows["bid_inr_per_click"].astype(float) * rows["relevance"].astype(float)).idxmax()
    else:
        raise ValueError(f"Unknown baseline: {method}")
    return rows.loc[idx].to_dict()
