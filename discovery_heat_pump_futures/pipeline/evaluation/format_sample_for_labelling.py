import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict

from discovery_heat_pump_futures import PROJECT_DIR
from discovery_heat_pump_futures.pipeline.evaluation.utils import (
    normalize_label,
    safe_parse_list,
)

DATA_INPUT_PATH = PROJECT_DIR / "inputs/balanced_sample.csv"
DATA_OUTPUT_PATH = PROJECT_DIR / "inputs/balanced_sample_formatted.csv"
MULTILABEL_COLS = [
    "trad_components",
    "non_trad_technologies",
    "system_design",
    "circular_economy",
]


def binarize_and_add_cols_for_human_labelling(
    df: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    """
    One-hot encodes a multi-label column into binary columns and adds corresponding
    empty columns for manual human annotation.

    Args:
        df: The input DataFrame containing the target column.
        target_col: The name of the column to binarize and annotate.

    Returns:
        A new DataFrame with one-hot encoded binary columns and matching _human columns added.
    """
    prefix = target_col

    # Parse + normalize
    df[target_col] = df[target_col].apply(
        lambda x: [normalize_label(i) for i in safe_parse_list(x)]
    )

    # One-hot encode with MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    binary = mlb.fit_transform(df[target_col])

    # Create dataframe with alternating prediction and human columns
    binary_cols = [f"{prefix}_{cls}" for cls in mlb.classes_]
    binary_df = pd.DataFrame(binary, columns=binary_cols)

    # Add corresponding _human columns (blank for now)
    for col in binary_cols:
        binary_df[f"{col}_human"] = pd.NA

    # Reorder: alternate prediction, human
    ordered_cols = []
    for col in binary_cols:
        ordered_cols.extend([col, f"{col}_human"])

    binary_df = binary_df[ordered_cols]
    df_final = pd.concat([df, binary_df], axis=1)

    return df_final


def merge_outputs(outputs: Dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges multiple DataFrames containing new binary and human columns into the original DataFrame.

    Args:
        outputs: A dictionary where each value is a DataFrame containing new columns for a target variable.
        df: The original base DataFrame to merge into.

    Returns:
        A DataFrame containing all original columns and the newly added columns from each output.
    """
    merged_df = df.copy()

    # Add only new columns from each output
    for _col, df_out in outputs.items():
        # Identify new columns added during binarization
        new_cols = [c for c in df_out.columns if c not in df.columns]

        # Join just those new columns
        merged_df = pd.concat([merged_df, df_out[new_cols]], axis=1)

    return merged_df


if __name__ == "__main__":
    df = pd.read_csv(DATA_INPUT_PATH)
    outputs = {}

    for col in MULTILABEL_COLS:
        print(f"Binarizing {col}...")
        df_final = binarize_and_add_cols_for_human_labelling(df, col)
        outputs[col] = df_final

    merged_df = merge_outputs(outputs, df)

    # Insert 'application_type_human' after 'application_type'
    idx = merged_df.columns.get_loc("application_type") + 1
    merged_df.insert(loc=idx, column="application_type_human", value=pd.NA)

    # Insert 'technology_readiness_level_human' after 'technology_readiness_level'
    idx = merged_df.columns.get_loc("technology_readiness_level") + 1
    merged_df.insert(loc=idx, column="technology_readiness_level_human", value=pd.NA)

    merged_df.to_csv(DATA_OUTPUT_PATH, index=False)
