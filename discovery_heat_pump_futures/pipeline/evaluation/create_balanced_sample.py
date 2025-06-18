from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List

from discovery_heat_pump_futures import PROJECT_DIR, logging
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
N_PER_LABEL = 5  # ie you will get 5 positive and 5 negative examples per subcategory


def make_balanced_sample(
    df: pd.DataFrame, target_columns: List[str], n_per_class: int = 5
) -> pd.DataFrame:
    """
    Create a balanced, deduplicated sample of rows from a DataFrame by sampling
    `n_per_class` positive and negative examples for each label in the target columns.

    Prints the number of positive and negative examples sampled per label.

    Args:
        df: The input DataFrame.
        target_columns: A list of column names to sample labels from.
        n_per_class: The number of positive and negative rows to sample per label. Defaults to 5.

    Returns:
        A deduplicated DataFrame containing the sampled rows.
    """
    sample_indices = set()
    summary = defaultdict(lambda: {"positive": 0, "negative": 0})

    for col in target_columns:
        # Detect and parse multi-label or single-label column
        is_multilabel = df[col].apply(
            lambda x: isinstance(x, str) and x.startswith("[")
        )

        if is_multilabel.any():
            parsed_col = df[col].apply(
                lambda x: [normalize_label(i) for i in safe_parse_list(x)]
            )
        else:
            parsed_col = df[col].apply(
                lambda x: [normalize_label(x)] if pd.notna(x) else []
            )

        # Binarize the labels
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(parsed_col)
        binary_df = pd.DataFrame(binary_matrix, columns=mlb.classes_, index=df.index)

        # For each label (i.e., binary column), sample positives and negatives
        for label in binary_df.columns:
            pos_indices = binary_df[binary_df[label] == 1].index
            neg_indices = binary_df[binary_df[label] == 0].index

            pos_sample = (
                pos_indices.to_series()
                .sample(n=min(n_per_class, len(pos_indices)), random_state=42)
                .tolist()
            )
            neg_sample = (
                neg_indices.to_series()
                .sample(n=min(n_per_class, len(neg_indices)), random_state=42)
                .tolist()
            )

            sample_indices.update(pos_sample + neg_sample)

            summary[(col, label)]["positive"] += len(pos_sample)
            summary[(col, label)]["negative"] += len(neg_sample)

    # Print summary
    print("\nSampling Summary:")
    for (col, label), counts in summary.items():
        print(
            f"Column: {col:<25} | Label: {label:<30} | +{counts['positive']} | -{counts['negative']}"
        )

    return df.loc[list(sample_indices)].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(DATA_INPUT_PATH)

    sampled_df = make_balanced_sample(df, TARGET_COLS, n_per_class=N_PER_LABEL)
    logging.info(f"Number of rows in balanced sample: {len(sampled_df)}")

    sampled_df.to_csv(DATA_OUTPUT_PATH, index=False)
