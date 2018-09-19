from io import StringIO

import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class CSVDataImportPipe(PipeBase):
    """
    Imports data from CSV and creates a dataframe using pd.read_csv().

    Args:
        filepath (str): path to read file
        content (str): raw data to import
        sep (bool): separation character to use
        engine (str): engine to be used, default is "python"
        index_col (str): column to use as index

    Returns:
        A dataframe with index_col as index column.
    """

    input_keys = ()
    output_keys = ("df",)

    def __init__(
        self,
        file_path: str = None,
        content: str = None,
        sep: bool = None,
        engine: str = "python",
        index_col: str = None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.content = content
        self.sep = sep
        self.engine = engine
        self.index_col = index_col

    def transform(self, data: Data, params: Params) -> Data:
        content = params.get("content") or self.content
        if content:
            content = StringIO(content)
        file_path = params.get("file_path") or self.file_path
        df = pd.read_csv(
            content or file_path,
            sep=params.get("sep") or self.sep,
            engine=params.get("engine") or self.engine,
            index_col=params.get("index_col") or self.index_col,
        ).sort_index(axis=1)
        return {"df": df}


class CSVDataExportPipe(PipeBase):
    """
    Exports a dataframe to CSV.
    Takes as input filepath (str), sep (str).
    Returns a CSV file at the specified location.
    """

    input_keys = ("df",)
    output_keys = ()

    def __init__(self, file_path: str = None, sep: str = None, **kwargs) -> None:
        super().__init__()

        self.file_path = file_path
        self.sep = sep or ","
        self.kwargs = kwargs

    def transform(self, data: Data, params: Params) -> Data:
        data["df"].to_csv(
            params.get("file_path") or self.file_path,
            sep=params.get("sep") or self.sep,
            **self.kwargs
        )
        return {}
