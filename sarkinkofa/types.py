from cv2.typing import MatLike
from pydantic import BaseModel, ConfigDict


class SarkiResult(BaseModel):
    """
    Data class for storing license plate information.
    """

    img: MatLike | None
    boxes: list[tuple[int, int, int, int]]
    confs: list[float]
    cls: list[int]
    labels: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Data classes
class PlateDetection(BaseModel):
    """
    Data class for storing license plate information.
    """

    box: tuple[int, int, int, int]
    conf: float
    number: str | None = None


class VehicleDetection(BaseModel):
    """
    Data class for storing vehicle detection information.
    """

    box: tuple[int, int, int, int]
    conf: float
    label: str
    plates: list[PlateDetection] | None = None


class SarkiDetection(BaseModel):
    """
    Data class for storing vehicle detection information.
    """

    img: MatLike
    vehicles: list[VehicleDetection] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
