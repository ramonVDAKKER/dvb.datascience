from typing import List

import pandas as pd

from ..sub_pipe_base import SubPipelineBase
from ..transform import (
    CategoricalImpute,
    FilterFeatures,
    ImputeWithDummy,
    LabelBinarizerPipe,
    Union,
)


class MetadataPipeline(SubPipelineBase):
    """
    Read metadata and make some pipes for processing the data
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, file_path: str, remove_vars: List = None) -> None:
        super().__init__("union")

        self.remove_vars = remove_vars or []
        self.file_path = file_path  # path to the file with the metadata
        self.metadata = pd.read_csv(self.file_path, sep=None, engine="python")

        rows = [
            row
            for row in self.metadata.itertuples()
            if row.varName not in self.remove_vars
        ]

        self.pipeline.addPipe("union", Union(len(rows)))

        for idx, row in enumerate(rows):
            self.pipeline.addPipe(
                "filter_" + row.varName, FilterFeatures([row.varName])
            )
            self.pipeline._connect(  # pylint: disable=W0212
                "", "df", "filter_" + row.varName, "df"
            )

            if row.varType == "num" and row.impMethod in ["mean", "median"]:
                self.pipeline.addPipe(
                    "impute_" + row.varName, ImputeWithDummy(strategy=row.impMethod)
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "filter_" + row.varName, "df", "impute_" + row.varName, "df"
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "impute_" + row.varName, "df", "union", "df%s" % idx
                )

            if row.varType == "cat" and row.impMethod == "mode":
                self.pipeline.addPipe("impute_" + row.varName, CategoricalImpute())
                self.pipeline.addPipe(
                    "labelbinarizer_" + row.varName, LabelBinarizerPipe()
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "filter_" + row.varName, "df", "impute_" + row.varName, "df"
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "impute_" + row.varName, "df", "labelbinarizer_" + row.varName, "df"
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "labelbinarizer_" + row.varName, "df", "union", "df%s" % idx
                )

            elif row.varType == "cat":
                self.pipeline.addPipe(
                    "labelbinarizer_" + row.varName, LabelBinarizerPipe()
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "filter_" + row.varName, "df", "labelbinarizer_" + row.varName, "df"
                )
                self.pipeline._connect(  # pylint: disable=W0212
                    "labelbinarizer_" + row.varName, "df", "union", "df%s" % idx
                )
