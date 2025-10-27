import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class BuildFeatures:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.processed_path = self.config["data"]["processed_path"]


    def split_data(self):
        logging.info("Splitting dataset into train/test sets...")
        df = pd.read_csv(self.processed_path)

        X = df.drop(columns="Stage")
        y = df["Stage"] - 1  # XGBoost expects 0-based labels

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["split"]["test_size"],
            random_state=self.config["split"]["random_state"],
            shuffle=True,
        )

        logging.info("Train/test split complete.")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    feature_builder = BuildFeatures()
    feature_builder.split_data()
