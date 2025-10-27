import pandas as pd
from pathlib import Path

# from fastapi.testclient import TestClient


# data validation
def test_data_integrity():
    path = Path("data/processed/train.csv")
    assert path.exists(), "processed dataset not found."
    df = pd.read_csv(path)

    # sanity ranges
    assert df["age"].between(18, 100).all(), "invalid age values."
    assert df["bilirubin"].between(0.1, 20).all(), "invalid bilirubin values."
    assert df["albumin"].between(1.5, 6).all(), "invalid albumin values."
    assert df["platelets"].between(20, 800).all(), "invalid platelet count."
    assert df["inr"].between(0.5, 4).all(), "invalid INR values."


def test_model_predicts():
    model_path = Path("models/model.pkl")
    assert model_path.exists(), "model file not found."
