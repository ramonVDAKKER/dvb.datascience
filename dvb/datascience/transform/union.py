from typing import Tuple, Set

import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class Union(PipeBase):
    """
    Merge the result of different Pipes. Merge can be done based on columns (default, axis=1) or
    rows (axis=0).
    When columns are merged, it's possible that columns are present in more than one input dataframe. At
    default, the second occurence of the column will be renamed with an underscore as suffix. Optional,
    duplicated colums are removed.

    The input_keys are generated at initialisation based on the
    the number of dfs, like:

    input_keys = ('df0', 'df1', 'df2', ...)

    """

    input_keys = ()  # type: Tuple
    output_keys = ("df",)

    def __init__(
        self,
        number_of_dfs,
        join: str = "outer",
        axis=1,
        remove_duplicated_columns: bool = False,
    ) -> None:
        super().__init__()

        self.input_keys = tuple("df%s" % i for i in range(0, number_of_dfs))
        self.join = join
        self.axis = axis
        self.remove_duplicated_columns = remove_duplicated_columns

    def transform(self, data: Data, params: Params) -> Data:
        dfs = []

        if self.axis == 1:
            found_column_names = set()  # type: Set[str]
            if self.remove_duplicated_columns:

                for df in data.values():
                    df = df.copy()
                    duplicated_feautures = found_column_names & set(df.columns)
                    df.drop(duplicated_feautures, axis=1)
                    found_column_names |= set(df.columns)
                    dfs.append(df)
            else:

                def check_name(name):
                    while name in found_column_names:
                        name = name + "_"
                    found_column_names.add(name)
                    return name

                for df in data.values():
                    dfs.append(df.rename(check_name, axis="columns"))
        else:
            dfs = [df.copy() for df in data.values()]

        return {"df": pd.concat(dfs, axis=self.axis, join=self.join)}
