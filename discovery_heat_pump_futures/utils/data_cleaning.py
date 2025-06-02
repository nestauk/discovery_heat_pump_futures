import pandas as pd


def format_title_abstract(
    df: pd.DataFrame, title: str = "title", abstract: str = "abstract"
) -> pd.DataFrame:
    """Clean title and abstract and merge them into a single column."""
    df = df.copy()
    df["title_abstract"] = (
        "TITLE: "
        + df[title].fillna("").str.lower()
        + " ABSTRACT: "
        + df[abstract].fillna("").str.lower()
    )
    return df
