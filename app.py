"""Streamlit demo for a RAG-based sponsored-ad auction."""
import streamlit as st
from src.retriever import AdRetriever
from src.auction import run_auction
from src.generator import compose_response

st.set_page_config(page_title="LLM Ad Auction Demo", page_icon="🎯", layout="wide")
st.title("RAG-Based Ad Auction for an LLM")
st.caption("Educational prototype: retrieve relevant ads, run a randomized auction, and disclose the sponsored result.")

@st.cache_resource
def load_retriever():
    return AdRetriever()

retriever = load_retriever()
query = st.text_input("User query", "Suggest a good machine learning laptop under ₹80,000")
top_k = st.slider("Number of retrieved candidates", 2, 8, 5)
seed = st.number_input("Random seed (change it to rerun the auction)", min_value=0, value=11, step=1)

if st.button("Retrieve and run auction", type="primary"):
    candidates = retriever.retrieve(query, top_k=top_k)
    import numpy as np
    result = run_auction(candidates, rng=np.random.default_rng(int(seed)))
    st.subheader("Retrieved advertisements")
    st.dataframe(candidates[["advertiser_id","product","bid_inr_per_click","relevance","price_inr"]], use_container_width=True)
    st.subheader("Auction ranking")
    st.dataframe(result["results"][["advertiser_id","product","relevance","bid_inr_per_click","gumbel_noise","auction_score","is_winner","payment_inr_per_click"]], use_container_width=True)
    winner = result["winner"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Winning advertiser", winner["advertiser_id"])
    c2.metric("Threshold payment / click", f"₹{result['payment']:.2f}")
    c3.metric("Winner relevance", f"{float(winner['relevance']):.3f}")
    st.subheader("LLM response (template-based)")
    st.write(compose_response(query, winner))
    st.info("Sponsored content is explicitly disclosed. This prototype uses a template generator, not an actual LLM.")
else:
    st.write("Enter a query and click **Retrieve and run auction**.")
    if hasattr(retriever, "fallback_reason"):
        st.warning("Sentence-transformer model unavailable; lexical TF-IDF retrieval is being used. "
                   "Install/download the embedding model for semantic retrieval.")
