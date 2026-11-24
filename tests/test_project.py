import unittest
import pandas as pd
from src.features import parse_budget, explicit_constraints
from src.retriever import AdRetriever
from src.pipeline import AdAuctionPipeline

class ProjectTests(unittest.TestCase):
    def test_budget_parser(self):
        self.assertEqual(parse_budget("laptop under ₹80,000"), 80000)
        self.assertEqual(parse_budget("phone under 1 lakh"), 100000)
        self.assertIsNone(parse_budget("good laptop for students"))

    def test_explicit_spec_parser(self):
        self.assertEqual(explicit_constraints("laptop with 16GB RAM and 512GB SSD"),
                         {"ram_gb": 16, "storage_gb": 512})

    def test_empty_hard_filter_returns_no_candidates(self):
        retriever = AdRetriever()
        candidates, _ = retriever.retrieve("laptop under ₹10000 with 64GB RAM")
        self.assertTrue(candidates.empty)

    def test_pipeline_returns_ctr_and_auction_winner(self):
        result = AdAuctionPipeline().run("gaming laptop under ₹80000 with 16GB RAM", seed=7)
        self.assertIsNotNone(result["winner"])
        self.assertIn("ctr_estimate", result["winner"])
        self.assertAlmostEqual(result["winner"]["relevance"], result["winner"]["ctr_estimate"])

if __name__ == "__main__":
    unittest.main()
