import pandas as pd
from imblearn.over_sampling import SMOTE

from ..classification_pipe_base import ClassificationPipeBase, Data, Params


class SMOTESampler(ClassificationPipeBase):
    """
    Resample the dataset.

    Note: the new df will not the indexes, because of extra creates row, the indexes
    won't be unique anymore.
    """

    input_keys = ("df", "df_metadata")
    output_keys = ("df",)

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs  # kwargs for SMOTE
        if not "random_state" in self.kwargs:
            self.kwargs["random_state"] = 149

    def fit(self, data: Data, params: Params):
        self._set_classification_labels(data["df"], data["df_metadata"])

    def transform(self, data: Data, params: Params) -> Data:
        self._set_classification_data(data["df"], data["df_metadata"])

        smote = SMOTE(**self.kwargs)

        new_X, new_y_true = smote.fit_sample(self.X, self.y_true)
        new_df = pd.DataFrame(new_X, columns=self.X.columns)
        new_df[self.y_true_label] = new_y_true

        return {"df": new_df}
