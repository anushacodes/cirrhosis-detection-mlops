import joblib
import pandas as pd
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def pipe_function(df):
    df = df.copy()

    mappings = {
        "Sex": {"F": 1, "M": 0},
        "Drug": {"Placebo": 1, "D-penicillamine": 0},
        "Hepatomegaly": {"Y": 1, "N": 0},
        "Spiders": {"Y": 1, "N": 0},
        "Edema": {"Y": 1, "N": 0, "S": 2},
        "Status": {"C": 0, "D": 1, "CL": 2},
    }

    drop_cols = ["N_Days", "Copper", "Cholesterol", "Tryglicerides", "Ascites"]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    expected_order = [
        "Stage",
        "Status",
        "Drug",
        "Age",
        "Sex",
        "Hepatomegaly",
        "Spiders",
        "Edema",
        "Bilirubin",
        "Albumin",
        "Alk_Phos",
        "SGOT",
        "Platelets",
        "Prothrombin",
    ]
    df = df.reindex(columns=[c for c in expected_order if c in df.columns])

    return df


class MakeDataset:

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_path = self.config["data"]["raw_path"]
        self.processed_path = self.config["data"]["processed_path"]

    def create_pipeline(self):
        joblib.dump(pipe_function, "models/processing.pkl")
        logging.info("preprocessor saved at models/processing.pkl")

        return pipe_function

    def clean(self):
        df = pd.read_csv(self.raw_path)
        pipeline = self.create_pipeline()

        df_processed = pipeline(df)
        df_processed.to_csv(self.processed_path, index=False)
        logging.info("Processed data saved to %s", self.processed_path)
        return df_processed


if __name__ == "__main__":
    make_data = MakeDataset()
    make_data.create_pipeline()
    make_data.clean()
