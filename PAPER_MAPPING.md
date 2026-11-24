# Mapping to “Ad Auctions for LLMs via Retrieval Augmented Generation”

| Project component | Role in this implementation | Relation to paper |
|---|---|---|
| Query/product retrieval | Produces query-ad similarity and candidate set | Retrieval supplies ad relevance information |
| CTR predictor + sigmoid calibration | Estimates calibrated expected click probability | Operationalizes the calibrated-retriever assumption that CTR is proportional to retrieval relevance `q_i` |
| Auction relevance | `relevance = calibrated_ctr` | Uses the calibrated CTR estimate as `q_i`; it is not multiplied by similarity a second time |
| Bid and Gumbel perturbation | `score = q_i * bid_i * exp(epsilon_i)` | Demonstration of randomized relevance-weighted allocation |
| Payment | Critical-bid style threshold against sampled runner-up | Simplified implementation; formal incentive properties require proof against the exact paper mechanism and assumptions |
| Synthetic interactions | Simulated impression/click labels | Engineering substitute only; not empirical CTR evidence from the paper or real users |

The paper does not mandate logistic regression or sigmoid calibration. These are practical choices for training and calibrating a demo retriever. In a real deployment, fit calibration on genuine impression/click data and address position bias.
