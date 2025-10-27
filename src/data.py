import joblib
import pandas as pd
import logging
import yaml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def pipe_function(df):
    """Callable wrapper for cleaning data."""
    df = df.copy()
    mappings = {
        "Sex": {"F": 1, "M": 0},
        "Drug": {"Placebo": 1, "D-penicillamine": 0},
        "Ascites": {"Y": 1, "N": 0},
        "Hepatomegaly": {"Y": 1, "N": 0},
        "Spiders": {"Y": 1, "N": 0},
        "Edema": {"Y": 1, "N": 0, "S": 2},
        "Status": {"C": 0, "D": 1, "CL": 2},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    if "Ascites" in df.columns:
        df.drop(columns=["Ascites"], inplace=True)

    return df


class MakeDataset:

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_path = self.config["data"]["raw_path"]
        self.processed_path = self.config["data"]["processed_path"]

    def create_pppipeline(self):
        joblib.dump(pipe_function, "models/processing.pkl")
        logging.info("preprocessor saved at models/processing.pkl")

        return pipe_function

    def clean(self):
        df = pd.read_csv(self.raw_path)
        pipeline = self.create_pppipeline()

        df_processed = pipeline(df)
        df_processed.to_csv(self.processed_path, index=False)
        logging.info("Processed data saved to %s", self.processed_path)
        return df_processed

if __name__ == "__main__":
    make_data = MakeDataset()
    make_data.create_pppipeline()
    make_data.clean()