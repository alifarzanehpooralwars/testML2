"""
Data loading utility:
    it loads the abstracts into a pandas dataframe (ids and abstracts) and check whether data is OK or not.
    Has only 1 function:
        load_abstracts()
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from .config import ABSTRACTS_FILE


def load_abstracts(path: Path = ABSTRACTS_FILE, n_rows : int | None = None) -> pd.DataFrame:
    """
    Parameters
        path:
            path to the jsonl data file
        n_rows: if provided, returns only the first n_rows rows;
    Returns:
        pd.DataFrame with columns of ["id", "abstract"]
    """

    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    df = pd.read_json(path, lines=True)

    if "id" not in df.columns or "abstract" not in df.columns:
        raise ValueError("column 'id' and 'abstract' are required")

    if n_rows:
        return df.iloc[:n_rows].copy()

    return df


if __name__ == "__main__":
    df = load_abstracts(n_rows = 6)
    print(df)

