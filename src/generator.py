"""Transparent response composer.

This deliberately uses a template so the project works without an LLM API key.
Replace this module with an LLM call if an approved provider/model is available.
"""
def compose_response(query, winner):
    if winner is None:
        return "I couldn't find a suitable sponsored recommendation."
    return (
        f"For your query, \"{query}\", one sponsored option is "
        f"{winner['product']} (₹{float(winner['price_inr']):,.0f}). "
        f"It is marked as a sponsored recommendation. "
        f"{winner['description']}. Please compare specifications and independent "
        f"reviews before purchasing."
    )
