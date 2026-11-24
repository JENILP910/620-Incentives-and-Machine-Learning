"""Transparent, auditable response composer (no external LLM required)."""
def compose_response(query, winner):
    if winner is None:
        return "I couldn't find a product matching the explicit constraints in your query. Try relaxing the budget or specifications."
    return (f'For "{query}", the sponsored recommendation is {winner["product"]} '
            f'(₹{float(winner["price_inr"]):,.0f}). Its estimated click-through rate '
            f'is {float(winner.get("ctr_estimate", winner.get("relevance", 0))):.2%} '
            f'from the demo CTR model. {winner["description"]} '
            "This is a sponsored recommendation; compare specifications and independent reviews before purchasing.")
