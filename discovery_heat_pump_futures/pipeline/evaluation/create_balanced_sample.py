import pandas as pd
from typing import List

from discovery_heat_pump_futures import PROJECT_DIR
from discovery_heat_pump_futures.pipeline.evaluation.utils import (
    normalize_label,
    safe_parse_list,
)

DATA_INPUT_PATH = PROJECT_DIR / "inputs/heat_pump_data_w_categories.csv"
DATA_OUTPUT_PATH = PROJECT_DIR / "inputs/balanced_sample.csv"
TARGET_COLS = [
    "application_type",
    "trad_components",
    "non_trad_technologies",
    "system_design",
    "circular_economy",
    "technology_readiness_level",
]
N_PER_LABEL = 10


def make_balanced_sample(
    df: pd.DataFrame, target_columns: List[str], n_per_class: int = 10
) -> pd.DataFrame:
    """
    Create a balanced, deduplicated sample of rows from a DataFrame by sampling up to
    `n_per_class` examples per label within each of the specified target columns.

    Multi-label columns are parsed and exploded to handle each label separately.

    Args:
        df: The input DataFrame.
        target_columns: A list of column names to sample labels from. Columns can be single- or multi-label.
        n_per_class: The number of rows to sample per unique label value. Defaults to 10.

    Returns:
        A deduplicated DataFrame containing the sampled rows.
    """
    sample_indices = set()

    for col in target_columns:
        # We assume it's multilabel if it contains a list
        is_multilabel = df[col].apply(
            lambda x: isinstance(x, str) and x.startswith("[")
        )

        if is_multilabel.any():
            # Multi-label column
            parsed = df[col].apply(
                lambda x: [normalize_label(i) for i in safe_parse_list(x)]
            )
        else:
            # Single-label column
            parsed = df[col].apply(
                lambda x: [normalize_label(x)] if pd.notna(x) else []
            )

        # Make one row per label (rather than all labels in a list)
        exploded = df.copy()
        exploded[col + "_norm"] = parsed
        exploded = exploded.explode(col + "_norm")

        # get up to n_per_class examples per label
        for label in exploded[col + "_norm"].dropna().unique():
            label_rows = exploded[exploded[col + "_norm"] == label]
            sampled = label_rows.sample(
                n=min(n_per_class, len(label_rows)), random_state=42
            )
            # only add it to our list of samples if it is not a duplicate
            sample_indices.update(sampled.index)

    return df.loc[list(sample_indices)].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(DATA_INPUT_PATH)

    sampled_df = make_balanced_sample(df, TARGET_COLS, n_per_class=N_PER_LABEL)

    sampled_df.to_csv(DATA_OUTPUT_PATH, index=False)
