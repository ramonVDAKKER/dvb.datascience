from sklearn.neighbors import KNeighborsClassifier

import dvb.datascience as ds


def run():
    p = ds.Pipeline()
    p.addPipe("read", ds.data.SampleData(dataset_name="breast_cancer"))
    p.addPipe(
        "split",
        ds.transform.TrainTestSplit(test_size=0.3, random_state=42),
        [("read", "df", "df")],
    )
    p.addPipe(
        "clf",
        ds.predictor.SklearnClassifier(clf=KNeighborsClassifier, n_neighbors=3),
        [("split", "df", "df"), ("read", "df_metadata", "df_metadata")],
    )
    p.addPipe(
        "score",
        ds.score.ClassificationScore(["accuracy", "confusion_matrix"]),
        [
            ("clf", "predict", "predict"),
            ("clf", "predict_metadata", "predict_metadata"),
        ],
    )

    params = {"split": {"train": True}}

    p.fit_transform(transform_params=params, fit_params=params)

    params = {"split": {"train": False}}
    p.transform(transform_params=params)
