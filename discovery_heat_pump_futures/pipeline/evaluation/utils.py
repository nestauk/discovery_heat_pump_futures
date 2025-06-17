import ast
import pandas as pd
import re
from typing import Union, List


def normalize_label(label: str) -> str:
    """Normalize a label string by lowercasing and replacing non-alphanumeric characters
    with underscores.

    Useful for use with MLB (MultiLabelBinarizer) because
    category levels that contain e.g. '-' cannot be used as column
    names.

    Args:
        label: The input string to normalize.

    Returns:
        A normalized string.
    """
    return re.sub(r"\W+", "_", label.strip().lower())


def safe_parse_list(val: Union[str, list, None]) -> List[str]:
    """Safely parse stringified lists and ensure we get a list."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        else:
            return []
    except (ValueError, SyntaxError):
        return []
