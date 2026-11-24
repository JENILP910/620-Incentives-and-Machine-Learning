"""End-to-end query -> constraints -> CTR retrieval -> auction -> response."""
from .retriever import AdRetriever
from .auction import run_auction
from .generator import compose_response

class AdAuctionPipeline:
    def __init__(self, data_path=None):
        self.retriever = AdRetriever(data_path=data_path) if data_path else AdRetriever()

    def run(self, query, top_k=10, seed=42):
        candidates, filters = self.retriever.retrieve(query, top_k=top_k)
        if candidates.empty:
            return {"query": query, "filters": filters, "candidates": candidates,
                    "auction": None, "winner": None,
                    "response": compose_response(query, None)}
        import numpy as np
        auction = run_auction(candidates, rng=np.random.default_rng(seed))
        winner = auction["winner"]
        return {"query": query, "filters": filters, "candidates": candidates,
                "auction": auction, "winner": winner,
                "response": compose_response(query, winner)}
