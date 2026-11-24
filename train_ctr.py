import argparse
from pathlib import Path
from src.ctr_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the calibrated CTR model")
    parser.add_argument("--interactions", default="data/interactions_synthetic.csv",
                        help="CSV with query, advertiser_id and binary click columns")
    args = parser.parse_args()
    metrics = train_model(Path(args.interactions))
    print("Held-out metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
