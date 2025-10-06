from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class TrainModel:

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config["model"]["params"]


    def train(self, X_train, y_train):

        logging.info("Training XGBoost model...")
        model = XGBClassifier(**self.model_params)
        model.fit(X_train, y_train)

        joblib.dump(model, "models/model.pkl")
        logging.info("Model saved to /models/model.pkl")

        return model
