import ast
import pandas as pd
from df2gspread import gspread2df as g2d
from oauth2client.service_account import ServiceAccountCredentials
import os

from discovery_heat_pump_futures import PROJECT_DIR


def get_credentials() -> ServiceAccountCredentials:
    """
    Load Google API credentials from a JSON keyfile for accessing Google Sheets and Drive.

    Returns:
        ServiceAccountCredentials: Credentials object for authenticating with Google APIs.
    """
    return ServiceAccountCredentials.from_json_keyfile_name(
        PROJECT_DIR / "credentials/discoverychilddevelopment-5233ca6bd42c.json",
        [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )


def get_taxonomy() -> pd.DataFrame:
    """
    Download and parse the taxonomy keyword sheet from Google Sheets using service account credentials.

    Returns:
        pd.DataFrame: A DataFrame containing the taxonomy, with the 'Keywords' column parsed from string to list.
    """
    credentials = get_credentials()

    keyword_df = g2d.download(
        os.environ.get("SHEET_ID_KEYWORDS"),
        "Keywords",
        credentials=credentials,
        col_names=True,
        row_names=True,
    )
    keyword_df["Keywords"] = keyword_df["Keywords"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    return keyword_df
