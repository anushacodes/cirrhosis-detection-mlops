import logging
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class PredictModel:

    def __init__(self, model_path="models/model.json"):

        logging.info("Loading XGBoost model from %s...", model_path)

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.pipeline = joblib.load("models/processing.pkl")

        logging.info("Model loaded for prediction.")

    def predict(self, X_test, y_test=None):
        X_test_processed = self.pipeline(X_test)
        preds = self.model.predict(X_test_processed)

        if y_test is not None:
            logging.info("Evaluating model performance...")

            acc = accuracy_score(y_test, preds)
            logging.info(f"Accuracy: {acc:.4f}")
            logging.info("\n" + classification_report(y_test, preds))

        return preds
