import argparse
from src.pipeline import AdAuctionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR-calibrated RAG ad auction demo")
    parser.add_argument("query", nargs="*", default=["gaming laptop under ₹80000 with 16GB RAM"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    query = " ".join(args.query)
    pipeline = AdAuctionPipeline()
    result = pipeline.run(query, top_k=args.top_k)
    print("\nQUERY:", result["query"])
    print("FILTERS:", result["filters"])
    if result["candidates"].empty:
        print("\nNo eligible ads found.")
    else:
        cols = [c for c in ["advertiser_id", "product", "price_inr", "bid_inr_per_click", "semantic_similarity", "ctr_estimate", "auction_score", "is_winner"] if c in result["auction"]["results"].columns]
        print("\nAUCTION CANDIDATES:")
        print(result["auction"]["results"][cols].to_string(index=False))
        print(f"\nPayment: ₹{result['auction']['payment']:.2f} per click")
    print("\nGENERATED RESPONSE:\n" + result["response"])
