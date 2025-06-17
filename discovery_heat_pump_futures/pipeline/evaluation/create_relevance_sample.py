import pandas as pd

from discovery_heat_pump_futures import PROJECT_DIR, logging

DATA_INPUT_PATH = PROJECT_DIR / "inputs/relevance_check.csv"
DATA_OUTPUT_PATH = PROJECT_DIR / "inputs/relevance_sample_for_labelling.csv"
CELL_SIZE = 10  # n per cell (positive vs negative, openalex vs patents)


def create_binary_sample(df: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a balanced sample of predicted positive
    and predicted negative cases for OpenAlex and Google Patents data.

    Args:
        df (pd.DataFrame): Dataframe with a column 'source' indicating whether the abstract is
        from OpenAlex or Google Patents, and a column 'is_heat_pump' indicating if the abstract is
        about heat pumps as classified by GPT.
        n (int, optional): The number of samples to take from each category (positive and negative).
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        pd.DataFrame: Dataframe balanced for 'source' and 'is_heat_pump'.
    """
    openalex_data = df[df["source"] == "openalex"]
    patents_data = df[df["source"] == "patents"]

    full_sample = pd.DataFrame()

    for source_data in [openalex_data, patents_data]:
        positives = source_data[source_data["is_heat_pump"] == "yes"]
        negatives = source_data[source_data["is_heat_pump"] == "no"]

        positives_sample = positives.sample(n=min(n, len(positives)), random_state=seed)
        negatives_sample = negatives.sample(n=min(n, len(negatives)), random_state=seed)
        sample_df = pd.concat([positives_sample, negatives_sample])

        full_sample = pd.concat([full_sample, sample_df])

    return full_sample


if __name__ == "__main__":
    data = pd.read_csv(DATA_INPUT_PATH)

    sample = create_binary_sample(data, n=CELL_SIZE, seed=42)

    logging.info(
        f"Sample distribution: {sample.groupby(['source', 'is_heat_pump']).size().unstack()}"
    )

    sample.to_csv(DATA_OUTPUT_PATH, index=False)
