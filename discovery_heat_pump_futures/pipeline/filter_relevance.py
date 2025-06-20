"""
Script to filter patents and abstracts, leaving only those that are actually about heat pumps.

Usage:
```
python discovery_heat_pump_futures/pipeline/filter_relevance.py
```
"""

import pandas as pd
from typing import List, Tuple

from discovery_utils.utils.llm import batch_check

from discovery_heat_pump_futures import PROJECT_DIR
from discovery_heat_pump_futures.utils.data_cleaning import format_title_abstract

RELEVANCE_SYSTEM = """
    You are evaluating if a document is PRIMARILY about heat pump technology.

    Heat pumps are devices that transfer heat from one place to another using a refrigeration cycle.
    They are used for heating, cooling, or both.

    Return JSON with:
    - "is_heat_pump": "yes" if primarily about heat pumps, "no" if just mentions pumps or is unrelated
    - "confidence": 0-1 (how confident you are in your answer)
    - "reason": brief explanation for your answer
    """

RELEVANCE_FIELDS = [
    {"name": "is_heat_pump", "type": "str", "description": "'yes' or 'no'"},
    {"name": "confidence", "type": "float", "description": "0-1 confidence score"},
    {
        "name": "reason",
        "type": "str",
        "description": "Brief explanation as to why you answered 'yes' or 'no'",
    },
]


def find_keywords(text: str, keyword_list: List[Tuple[str, str, str]]) -> List[str]:
    """
    Find keywords from a keyword list that appear in the input text.

    Args:
        text (str): The input text to search for keywords.
        keyword_list (List[Tuple[str, str, str]]): A list of tuples where each tuple contains
            (keyword, category, subcategory).

    Returns:
        List[str]: A list of keywords that were found in the input text.
    """
    text_lower = text.lower()
    matches = [kw for kw, _, _ in keyword_list if kw in text_lower]
    return matches


def filter_on_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Filters a DataFrame to only include rows where the 'title_abstract' column
    contains one or more predefined keywords from a taxonomy.

    Adds two new columns to the DataFrame:
        - 'matched_keywords': a list of matched keywords
        - 'n_keywords': the number of matched keywords

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain a 'title_abstract'
        column.

    Returns:
        pd.DataFrame: A filtered DataFrame including only rows where at least one
        keyword was found.
    """


if __name__ == "__main__":
    OPENALEX_URL = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/future_heat_pumps/heat_pumps_openalex.csv"
    PATENT_URL = "https://discovery-hub-open-data.s3.eu-west-2.amazonaws.com/future_heat_pumps/heat_pumps_patents.json"

    openalex_df = pd.read_csv(OPENALEX_URL, low_memory=False)
    patents_df = pd.read_json(PATENT_URL, lines=True)

    openalex_df = format_title_abstract(openalex_df)
    patents_df = format_title_abstract(patents_df)

    openalex_df["source"] = "openalex"
    patents_df["source"] = "patents"

    combined_df = pd.concat(
        [
            patents_df[["publication_number", "source", "title_abstract"]]
            .rename(columns={"publication_number": "id"}),

            openalex_df[
                [
                    "id",
                    "source",
                    "title_abstract",
                ]
            ],
        ],
        ignore_index=True,
    )

    relevance_proc = batch_check.LLMProcessor(
        model_name="gpt-4o-mini",
        temperature=0,
        system_message=RELEVANCE_SYSTEM,
        session_name="heat_pump_relevance",
        output_fields=RELEVANCE_FIELDS,
        output_path=str(PROJECT_DIR / "outputs/relevance_check.jsonl"),
    )

    ids = combined_df["id"].tolist()
    text = combined_df["title_abstract"].tolist()
    data_dict = dict(zip(ids, text, strict=True))

    relevance_proc.run(data_dict, batch_size=50, sleep_time=0.5)

    classified_data = pd.read_json(relevance_proc.output_path, lines=True)

    output_data = pd.merge(
        combined_df,
        classified_data[["id", "is_heat_pump", "confidence", "reason"]],
        on="id",
        how="left",
    )

    output_data.to_csv(PROJECT_DIR / "inputs/relevance_check.csv", index=False)

    filtered_data = output_data[output_data["is_heat_pump"] == "yes"].copy()
    filtered_data.to_csv(
        PROJECT_DIR / "inputs/heat_pump_relevant_data.csv", index=False
    )
