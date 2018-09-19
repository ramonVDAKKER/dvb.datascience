import sys
import warnings
from importlib import reload
import pathlib
import logging.config
from typing import Any, Optional

from IPython import get_ipython

from . import data
from . import eda
from . import predictor
from . import score
from . import transform
from .classification_pipe_base import ClassificationPipeBase
from .pipe_base import PipeBase
from .pipeline import Pipeline
from .sub_pipe_base import SubPipelineBase

import numpy as np
import random as rn


def load_module(name: str, disable_warnings: bool = True, random_seed:Optional[int]=1122) -> Any:
    """
    Convenience function for running an experiment. This function reloads the
    experiment when it is already loaded, so any changes in the [.. missing word ..] of that
    experiment will be used. Usage:

        import dvb.datascience as ds
        p = ds.load_module('experiment').run()

    `p` can be used to access the contents of the pipeline, like:

        p.get_pipe_output('predict')

    in case you define a 'run()' method in 'experiment.py' returning the pipeline object
    """

    if random_seed:
        np.random.seed(1122)
        rn.seed(1122)

    pathlib.Path("./log").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./tmp").mkdir(parents=True, exist_ok=True)

    if pathlib.Path("./logging.conf").exists():
        logging.config.fileConfig("logging.conf")

    if get_ipython():
        get_ipython().run_line_magic("matplotlib", "inline")

    if disable_warnings:
        warnings.filterwarnings("ignore")

    if name in sys.modules:
        return reload(sys.modules[name])

    return __import__(name)


def run_module(name: str, disable_warnings: bool = True) -> Any:
    warnings.warn("run_module is replaced by load_module",
                  PendingDeprecationWarning)
    return load_module(name, disable_warnings)
