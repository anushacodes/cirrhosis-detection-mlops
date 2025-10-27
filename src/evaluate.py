import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class PredictModel:

    def __init__(self, model_path="models/model.pkl"):
        self.model = joblib.load(model_path)
        logging.info("Model loaded for prediction.")

    def predict(self, X_test, y_test=None):
        logging.info("Generating predictions...")
        preds = self.model.predict(X_test)

        if y_test is not None:
            logging.info("Evaluating model performance...")
            acc = accuracy_score(y_test, preds)
            logging.info(f"Accuracy: {acc:.4f}")
            logging.info("\n" + classification_report(y_test, preds))

            conf_mat = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_mat, annot=True, fmt="d", cmap="coolwarm")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            # plt.savefig("reports/figures/confusion_matrix.png")

        return preds
