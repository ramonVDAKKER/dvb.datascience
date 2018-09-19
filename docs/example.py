# This file is used in the RunExample.ipynb notebook to show the usage of the pipeline from a notebook
# It allows for using the visual features of a notebook, as well as the ability to modify your code using
# An IDE (auto complete, version control, etc)

from IPython.core.display import display, HTML

import dvb.datascience as ds


# Enable this for having debug logging embedded in your jupyter notebook
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("test")


def run():
    display(HTML("<h1>Experiment started</h1>"))

    display(HTML("<h2>Running pipeline 1</h2>"))
    p = ds.Pipeline()
    p.addPipe("read", ds.data.CSVDataImportPipe())
    p.addPipe(
        "read2",
        ds.data.CSVDataImportPipe(),
        comment="Very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, long comment",
    )
    p.addPipe("numeric", ds.transform.FilterTypeFeatures(), [("read", "df", "df")])
    p.addPipe(
        "numeric2",
        ds.transform.FilterTypeFeatures(),
        [("read2", "df", "df")],
        comment="Very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, long comment",
    )
    p.addPipe(
        "boxplot", ds.eda.BoxPlot(), [("numeric", "df", "df"), ("numeric2", "df", "df")]
    )
    p.draw_design()

    display(HTML("<h2>Running pipeline 2</h2>"))
    from sklearn.neighbors import KNeighborsClassifier

    p = ds.Pipeline()
    p.addPipe(
        name="read",
        pipe=ds.data.SampleData("iris"),
        comment="Default sklearn iris dataset",
    )
    p.addPipe(
        "clf",
        ds.predictor.SklearnClassifier(KNeighborsClassifier, n_neighbors=3),
        [("read", "df", "df"), ("read", "df_metadata", "df_metadata")],
        comment="Short com.",
    )
    p.addPipe(
        "score",
        ds.score.ClassificationScore(),
        [
            ("clf", "predict", "predict"),
            ("clf", "predict_metadata", "predict_metadata"),
        ],
        comment="Very long comment to describe some very important stuff regarding this processing step",
    )
    p.fit_transform()

    display(HTML("<h1>Experiment done</h1>"))

    return p
