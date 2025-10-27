import pandas as pd
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


class MakeDataset:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_path = self.config["data"]["raw_path"]
        self.processed_path = self.config["data"]["processed_path"]

    def load_and_clean(self):

        logging.info("Loading raw dataset...")
        df = pd.read_csv(self.raw_path)

        logging.info("Encoding categorical variables...")

        df["Sex"] = df["Sex"].replace({"F": 1, "M": 0})
        df["Drug"] = df["Drug"].replace({"Placebo": 1, "D-penicillamine": 0})
        df["Ascites"] = df["Ascites"].replace({"Y": 1, "N": 0})
        df["Hepatomegaly"] = df["Hepatomegaly"].replace({"Y": 1, "N": 0})
        df["Spiders"] = df["Spiders"].replace({"Y": 1, "N": 0})
        df["Edema"] = df["Edema"].replace({"Y": 1, "N": 0, "S": 2})
        df["Status"] = df["Status"].replace({"C": 0, "D": 1, "CL": 2})

        df = df.drop(columns=["Ascites"])

        logging.info("Saving processed dataset...")
        df.to_csv(self.processed_path, index=False)
        logging.info(f"Processed data saved at {self.processed_path}")

        return df
