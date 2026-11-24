"""Streamlit web interface for the CTR-calibrated RAG ad auction demo."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.pipeline import AdAuctionPipeline

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="RAG Ad Auction | CTR Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading catalog and CTR model…")
def get_pipeline():
    return AdAuctionPipeline()


def money(value):
    return f"₹{float(value):,.2f}"


def show_filter_summary(filters):
    items = []
    if filters.get("category"):
        items.append(f"Category: **{filters['category']}**")
    if filters.get("budget_inr") is not None:
        items.append(f"Maximum price: **{money(filters['budget_inr'])}**")
    for key, value in filters.get("specifications", {}).items():
        label = {"ram_gb": "RAM", "storage_gb": "Storage"}.get(key, key)
        suffix = "GB" if key.endswith("_gb") else ""
        items.append(f"{label}: **{value}{suffix}+**")
    if items:
        st.markdown(" · ".join(items))
    else:
        st.caption("No hard constraints detected. Candidates are ranked using query-product relevance and estimated CTR.")


with st.sidebar:
    st.title("🎯 RAG Ad Auction")
    st.caption("Retrieval · CTR calibration · Randomized auction")
    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Parse explicit constraints\n"
        "2. Retrieve eligible products\n"
        "3. Estimate calibrated CTR\n"
        "4. Run relevance-weighted auction\n"
        "5. Show the sponsored result"
    )
    st.divider()
    st.caption("Course-project prototype. CTR training interactions are synthetic, not real customer activity.")

st.title("Ad Auctions for LLMs via Retrieval-Augmented Generation")
st.markdown(
    "Explore how query-aware retrieval and a calibrated click-through-rate estimate "
    "affect the allocation of a sponsored product recommendation."
)

examples = [
    "gaming laptop under ₹80000 with 16GB RAM",
    "good laptop for AI/ML",
    "smartphone for photography under ₹50000",
    "tablet for students under ₹30000",
    "headphones for online meetings",
]

with st.container(border=True):
    st.subheader("Try a search")
    selected_example = st.selectbox("Example queries", ["Choose an example…"] + examples)
    default_query = selected_example if selected_example != "Choose an example…" else "gaming laptop under ₹80000 with 16GB RAM"
    with st.form("query_form"):
        query = st.text_input("What are you looking for?", value=default_query, placeholder="e.g. laptop under ₹80,000 with 16GB RAM")
        left, middle, right = st.columns([1, 1, 2])
        with left:
            top_k = st.slider("Candidates", min_value=3, max_value=30, value=10)
        with middle:
            seed = st.number_input("Auction seed", min_value=0, max_value=999999, value=42, step=1)
        with right:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            run_clicked = st.form_submit_button("Run retrieval + auction", type="primary", use_container_width=True)

if run_clicked:
    if not query.strip():
        st.warning("Enter a product query to continue.")
    else:
        try:
            with st.spinner("Retrieving products, estimating CTR and running the auction…"):
                pipeline = get_pipeline()
                st.session_state["auction_result"] = pipeline.run(query.strip(), top_k=int(top_k), seed=int(seed))
                st.session_state["last_query"] = query.strip()
        except Exception as exc:
            st.error(f"Could not run the pipeline: {exc}")

result = st.session_state.get("auction_result")

if result:
    st.divider()
    st.header("Results")
    st.markdown(f"**Query:** {result['query']}")
    show_filter_summary(result["filters"])

    if result["candidates"].empty:
        st.warning("No products satisfy the explicit constraints. Try relaxing the budget or specifications.")
    else:
        winner = result["winner"]
        auction_rows = result["auction"]["results"].copy()
        winner_col, ctr_col, bid_col, payment_col = st.columns(4)
        winner_col.metric("Sponsored winner", winner["product"])
        ctr_col.metric("Estimated CTR", f"{winner['ctr_estimate']:.2%}")
        bid_col.metric("Advertiser bid", f"{money(winner['bid_inr_per_click'])}/click")
        payment_col.metric("Demo payment", f"{money(result['auction']['payment'])}/click")

        st.success(f"Winner: **{winner['product']}** — {money(winner['price_inr'])}")
        st.markdown("**Generated sponsored response**")
        st.info(result["response"])

        tab_auction, tab_candidates, tab_details = st.tabs(["Auction ranking", "Retrieved candidates", "Decision details"])

        with tab_auction:
            st.subheader("Auction score ranking")
            chart_df = auction_rows[["product", "auction_score"]].copy().set_index("product")
            st.bar_chart(chart_df, height=360)
            display_cols = ["product", "ctr_estimate", "bid_inr_per_click", "auction_score", "is_winner"]
            display = auction_rows[display_cols].copy()
            display["ctr_estimate"] = display["ctr_estimate"].map(lambda x: f"{x:.2%}")
            display["bid_inr_per_click"] = display["bid_inr_per_click"].map(money)
            display["auction_score"] = display["auction_score"].map(lambda x: f"{x:.5f}")
            display["is_winner"] = display["is_winner"].map(lambda x: "🏆 Winner" if x else "")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("Score = calibrated CTR relevance × bid × exp(Gumbel noise). The random seed makes this run reproducible.")

        with tab_candidates:
            st.subheader(f"Top {len(result['candidates'])} eligible products")
            candidates = result["candidates"][["advertiser_id", "product", "category", "price_inr", "bid_inr_per_click", "semantic_similarity", "ctr_estimate"]].copy()
            candidates = candidates.rename(columns={
                "advertiser_id": "Advertiser ID", "product": "Product", "category": "Category",
                "price_inr": "Price (INR)", "bid_inr_per_click": "Bid (INR/click)",
                "semantic_similarity": "Semantic similarity", "ctr_estimate": "Estimated CTR",
            })
            st.dataframe(
                candidates.style.format({
                    "Price (INR)": "₹{:,.0f}", "Bid (INR/click)": "₹{:.2f}",
                    "Semantic similarity": "{:.4f}", "Estimated CTR": "{:.2%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with tab_details:
            st.subheader("Why did this product win?")
            st.markdown(
                f"- **Retrieval:** The product was eligible under the detected constraints and ranked among the top candidates.\n"
                f"- **Semantic similarity:** `{float(winner['semantic_similarity']):.4f}`\n"
                f"- **Calibrated CTR proxy (qᵢ):** `{float(winner['ctr_estimate']):.4f}`\n"
                f"- **Bid:** `{money(winner['bid_inr_per_click'])}` per click\n"
                f"- **Perturbed auction score:** `{float(winner['auction_score']):.6f}`\n"
                f"- **Threshold-style demo payment:** `{money(result['auction']['payment'])}` per click"
            )
            st.caption("The displayed payment follows this project's simplified single-slot threshold-payment implementation; it is not a claim of a complete formal reproduction of the paper's mechanism.")

st.divider()
with st.expander("CTR model evaluation and data caveat"):
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Held-out ROC-AUC", f"{metrics['test_auc']:.3f}")
        m2.metric("Log loss", f"{metrics['test_log_loss']:.3f}")
        m3.metric("Brier score", f"{metrics['test_brier_score']:.3f}")
        m4.metric("Test impressions", f"{metrics['test_rows']:,}")
        st.caption(f"Training source: {metrics['training_data']} · {metrics['rows']:,} total impression rows")
    st.warning(
        "The catalog contains product details and bids, but no genuine click history. "
        "The CTR model is trained on simulated impression/click records. Its CTR values "
        "and evaluation metrics are demonstration results, not real-world customer CTRs."
    )
    st.markdown("The model uses query/product text signals and contextual features. Bid values are excluded from CTR prediction; bid enters only in the auction score.")

st.caption("Research prototype · Synthetic CTR data · Not a production advertising system")
