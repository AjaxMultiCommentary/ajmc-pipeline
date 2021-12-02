"""WORK IN PROGRESS."""

import pandas as pd
from commons.variables import METADATA_SPREADSHEET_ID, METADATA_WORKSHEET_NAME


def get_metadata_df() -> pd.DataFrame:
    URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
        METADATA_SPREADSHEET_ID,
        METADATA_WORKSHEET_NAME,
    )

    return pd.read_csv(URL)
