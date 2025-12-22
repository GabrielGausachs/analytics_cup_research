from .load_data import load_physical_data
from ..utils.preprocessing import preprocess_physical_data
import pandas as pd


def get_physical_data_processed(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess physical data for a given match.

    Args:
        data_path (str): The path to the data directory.

    Returns:
        pd.DataFrame: Preprocessed physical data DataFrame.
    """
    physical_data = load_physical_data(data_path)
    physical_data_processed = preprocess_physical_data(physical_data)

    return physical_data_processed