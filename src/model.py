from xgboost import XGBClassifier
import logging
import yaml
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class TrainModel:

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config["model"]["params"]

    def train(self, X_train, y_train):

        logging.info("Training XGBoost model...")

        pipeline = joblib.load("models/processing.pkl")
        X_train_processed = pipeline(X_train)

        model = XGBClassifier(**self.model_params)
        model.fit(X_train_processed, y_train)

        model.save_model("models/model.json")

        logging.info("Model saved to /models/model.json")

        return model
