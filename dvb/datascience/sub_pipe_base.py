from typing import Dict, Any

from .pipe_base import Data, Params, PipeBase
from .pipeline import Pipeline


class PassData(PipeBase):
    def __init__(self, subpipeline, output_keys):
        self.output_keys = output_keys
        self.subpipeline = subpipeline

    def transform(self, data: Data, params: Params) -> Data:
        return self.subpipeline.data_from_pipeline


class SubPipelineBase(PipeBase):
    def __init__(self, output_pipe_name: str) -> None:
        """
        Define a pipeline within a pipeline. The output of the pipe with the name `output_name` will
        be returned to the caller
        """
        super().__init__()

        self.pipeline = Pipeline()
        self.pipeline.addPipe("pass_data", PassData(self, self.input_keys))
        self.output_pipe_name = output_pipe_name

    def fit_transform(
        self, data: Data, transform_params: Params, fit_params: Params
    ) -> Data:
        self.data_from_pipeline = (
            data
        )  # store data to a location where PassData (first pipe of subpipeline) can pass the data to the next step
        self.pipeline.fit_transform(
            data=data, transform_params=transform_params, fit_params=fit_params
        )
        return self.pipeline.get_pipe_output(self.output_pipe_name)

    def transform(self, data: Data, params: Params) -> Data:
        self.pipeline.transform(data=data, transform_params=params)
        return self.pipeline.get_pipe_output(self.output_pipe_name)

    def load(self, state: Dict[str, Any]) -> None:
        """
        load all fitted attributes of this Pipe from `state`.
            """
        for pipe_name, pipe in self.pipeline.pipes.items():
            pipe.load(state[pipe_name])

    def save(self) -> Dict[str, Any]:
        """
        Return all fitted attributes of this Pipe in a Dict which is JSON serializable.
        """
        state = {}
        for pipe_name, pipe in self.pipeline.pipes.items():
            state[pipe_name] = pipe.save()

        return state
