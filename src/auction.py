"""Single-slot randomized relevance-weighted auction.

The calibrated CTR estimate is used as q_i (relevance). Do not multiply it by
semantic similarity again: similarity is already an input to the CTR model.
"""
import numpy as np

def run_auction(ads, rng=None):
    if len(ads) == 0:
        raise ValueError("Auction requires at least one candidate ad.")
    rng = rng or np.random.default_rng()
    rows = ads.copy().reset_index(drop=True)
    for col in ("bid_inr_per_click", "relevance"):
        if col not in rows:
            raise ValueError(f"Missing required column: {col}")
    bids = rows["bid_inr_per_click"].astype(float).to_numpy()
    q = rows["relevance"].astype(float).to_numpy()
    if np.any(bids < 0) or np.any(q <= 0):
        raise ValueError("Bids must be nonnegative and relevance scores positive.")
    eps = rng.gumbel(loc=0.0, scale=1.0, size=len(rows))
    scores = q * bids * np.exp(eps)
    winner_idx = int(np.argmax(scores))
    rows["gumbel_noise"] = eps
    rows["auction_score"] = scores
    rows["is_winner"] = False
    rows.loc[winner_idx, "is_winner"] = True
    if len(rows) == 1:
        payment, runner_idx = 0.0, None
    else:
        rival_scores = scores.copy(); rival_scores[winner_idx] = -np.inf
        runner_idx = int(np.argmax(rival_scores))
        payment = q[runner_idx] * bids[runner_idx] * np.exp(eps[runner_idx]) / (q[winner_idx] * np.exp(eps[winner_idx]))
        payment = float(np.clip(payment, 0.0, bids[winner_idx]))
    rows["payment_inr_per_click"] = 0.0
    rows.loc[winner_idx, "payment_inr_per_click"] = payment
    return {"winner": rows.loc[winner_idx].to_dict(),
            "runner_up": None if runner_idx is None else rows.loc[runner_idx].to_dict(),
            "payment": payment,
            "results": rows.sort_values("auction_score", ascending=False).reset_index(drop=True)}
