import abc
import codecs
import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


Data = Dict[str, Any]
Params = Dict[str, Any]


class PipeBase(metaclass=abc.ABCMeta):
    """
    Common base class for all pipes
    """

    input_keys = ("df",)  # type: Tuple[str, ...]
    output_keys = ("df",)  # type: Tuple[str, ...]

    name = None  # type: str

    fit_attributes = (
        tuple()
    )  # type: Sequence[Tuple[str, Optional[Union[str, Callable]], Optional[Union[str, Callable]]]]

    def __repr__(self):
        return f"Pipe({self.name!r})"

    def __init__(self):
        # a mapping from transform nr to a mapping with different kinds of data which are needed to store the transform
        # results for combining the results of multiple transforms in one output
        logger.info("Initiating pipe base")
        self.transform_data = defaultdict(dict)  # Dict[str, Dict[str, Any]]

    def get_transform_data_by_key(self, key: str) -> List[Any]:
        """
        Get all values for a certain key for all transforms
        """
        l = []
        for d in self.transform_data.values():
            if key in d:
                l.append(d[key])

        return l

    def fit_transform(
        self, data: Data, transform_params: Params, fit_params: Params
    ) -> Data:
        self.fit(data, fit_params)
        return self.transform(data, transform_params)

    def fit(self, data: Data, params: Params):
        """
        Train on a dataset `df` and store the learnings so `transform` can be called later on
        to transform based on the learnings.
        """
        pass

    @abc.abstractmethod
    def transform(self, data: Data, params: Params) -> Data:
        """
        Perform an operations on `df` using the kwargs and the learnings from training.
        Transform will return a tuple with the transformed dataset and some output.
        The transformed dataset will be the input for the next plumber.
        The output will be collected and shown to the user.
        """

    figs = None  # type: Dict[Any, plt.Figure]

    def get_fig(self, idx: Any):
        """
        Set in plt the figure to one to be used.

        When `idx` has already be used, it will set the same Figure
        so data can be added to that plot. Otherwise a new Figure will be set
        """
        if self.figs is None:
            self.figs = {}

        if idx not in self.figs:
            self.figs[idx] = plt.figure()

        fig = self.figs[idx]
        plt.figure(fig.number)
        return fig

    def load(self, state: Dict[str, Any]):
        """
        load all fitted attributes of this Pipe from `state`.

        Note:
        All PipeBase subclasses can define a `fit_attributes` attribute which contains
        a tuple for every attribute which is set during the fit phase. Those are
        the attributes which needs to be saved in order to be loaded in a new process
        without having to train (fit) the pipeline. This is useful ie for model inference.
        The tuple for every attribute consist of (name, serializer, deserializer).

        The (de)serializer are needed to convert to/from a JSON serializable format and can be:
        - None: No conversion needed, ie for str, int, float, list, bool
        - 'pickle': The attribute will be pickled and stored as base64, so it can be part of a json
        - callable: a function which will get the object to be (de)serialized and need to return the (de)serialized version
        """
        for cls in self.__class__.__mro__:
            fit_attributes = cls.__dict__.get(
                "fit_attributes", []
            )  # List[Tuple[str, Optional[Union[str, callable]], Optional[Union[str, callable]]]]
            for (name, _, deserializer) in fit_attributes:
                if name not in state:
                    continue
                if deserializer == "pickle":
                    deserializer = self._string_base64_pickle
                value = (
                    state[name] if deserializer is None else deserializer(state[name])
                )
                setattr(self, name, value)

    def save(self) -> Dict[str, Any]:
        """
        Return all fitted attributes of this Pipe in a Dict which is JSON serializable.
        """
        state = {}
        for cls in self.__class__.__mro__:
            fit_attributes = cls.__dict__.get(
                "fit_attributes", []
            )  # List[Tuple[str, Optional[Union[str, callable]], Optional[Union[str, callable]]]]
            for (name, serializer, _) in fit_attributes:
                if serializer == "pickle":
                    serializer = self._pickle_base64_stringify
                value = getattr(self, name)
                state[name] = value if serializer is None else serializer(value)

        return state

    @staticmethod
    def _pickle_base64_stringify(obj: Any) -> str:
        return codecs.encode(pickle.dumps(obj), "base64").decode()

    @staticmethod
    def _string_base64_pickle(obj: str) -> Any:
        return pickle.loads(codecs.decode(obj.encode(), "base64"))
