"""Domain exceptions for the training pipeline.

The pipeline used to raise HTTPException from inside a BackgroundTask, where
the status code is meaningless — nothing is going to turn it into a response.
These carry the same information as a plain exception the job runner can
classify, and the routes translate them at the edge where a status code is
real.
"""


class PipelineError(Exception):
    """Base class for anything the training pipeline rejects."""

    #: Grouping the UI uses to choose its wording.
    kind = "pipeline"
    #: What the route should return if this surfaces during a request.
    status_code = 400


class DatasetError(PipelineError):
    """The uploaded data cannot be used at all."""

    kind = "dataset"


class DatasetTooSmallError(DatasetError):
    pass


class DatasetTooLargeError(DatasetError):
    status_code = 413


class TargetValidationError(PipelineError):
    """The chosen target column cannot be learned. Message is user-facing."""

    kind = "target"


class TrainingError(PipelineError):
    """Every candidate model failed."""

    kind = "training"


class ArtifactError(PipelineError):
    """Stored artifacts are missing or from an incompatible pipeline version."""

    kind = "artifact"
    status_code = 409
