import logging
from src.data import MakeDataset
from src.features import BuildFeatures
from src.model import TrainModel
from src.evaluate import PredictModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == "__main__":
    logging.info("Starting pipeline...")

    # Step 1: Data preparation
    make_data = MakeDataset()
    df = make_data.load_and_clean()

    # Step 2: Feature splitting
    feature_builder = BuildFeatures()
    X_train, X_test, y_train, y_test = feature_builder.split_data()

    # Step 3: Model training
    trainer = TrainModel()
    model = trainer.train(X_train, y_train)

    # Step 4: Model evaluation + predictions
    predictor = PredictModel()
    preds = predictor.predict(X_test, y_test)

    logging.info("Pipeline completed successfully.")
