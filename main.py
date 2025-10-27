import logging
from src.data import MakeDataset
from src.features import BuildFeatures
from src.model import TrainModel
from src.evaluate import PredictModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


if __name__ == "__main__":

    logging.info("Starting pipeline...")

    make_data = MakeDataset()
    make_data.create_pipeline()
    df = make_data.clean()

    feature_builder = BuildFeatures()
    X_train, X_test, y_train, y_test = feature_builder.split_data()

    trainer = TrainModel()
    model = trainer.train(X_train, y_train)

    predictor = PredictModel()
    preds = predictor.predict(X_test, y_test)

    logging.info("Pipeline completed successfully.")
