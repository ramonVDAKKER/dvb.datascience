import pandas as pd
from IPython.core.display import display

from ..pipe_base import Data, Params
from .base import AnalyticsBase


class Dump(AnalyticsBase):
    """
    Dump the read data

    Args:
        data: Dataframe to be used

    Returns:
        Empty dataframe.
    """

    input_keys = ("df",)
    output_keys = ("output",)

    def transform(self, data: Data, params: Params) -> Data:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(data["df"])
        return {"output": data["df"]}
