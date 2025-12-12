import pandas as pd
import os
import logging
from pathlib import Path


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_external_data(train_path: Path, test_path: Path):

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.debug(f"Loaded external train: {train_df.shape}")
        logger.debug(f"Loaded external test: {test_df.shape}")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading external CSV files: {e}")
        raise


def minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
     
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logger.debug(f"Removed {before - after} duplicate rows.")
    return df

def save_raw(train_df: pd.DataFrame, test_df: pd.DataFrame, raw_dir: Path):
    
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(raw_dir / "train.csv", index=False)
        test_df.to_csv(raw_dir / "test.csv", index=False)

        logger.debug(f"Saved raw train & test to: {raw_dir}")
    except Exception as e:
        logger.error(f"Error saving raw data: {e}")
        raise



def main():
    try:
        root = Path(__file__).parent.parent.parent

        external_train = root / "data" / "external" / "fraudTrain.csv"
        external_test = root / "data" / "external" / "fraudTest.csv"
        raw_dir = root / "data" / "raw"

        logger.debug(f"External train path: {external_train}")
        logger.debug(f"External test path: {external_test}")

        train_df, test_df = load_external_data(external_train, external_test)

        train_df = minimal_clean(train_df)
        test_df = minimal_clean(test_df)
       
        save_raw(train_df, test_df, raw_dir)

        logger.debug("Data ingestion completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
