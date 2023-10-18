import numpy as np
from dataclasses import dataclass


# Data classes
@dataclass
class LicensePlateDetection:
    """
    Data class for storing license plate information.
    """

    img: np.ndarray
    box: np.ndarray
    conf: float
    cls: str
    color: list[int]
    number: str | None = None
    number_conf: float | None = None

    def to_dict(self) -> dict:
        """
        Converts the license plate detection data class to a dictionary.

        Returns:
            dict: The license plate detection data class as a dictionary.
        """
        return {
            "box": self.box.astype(int).tolist(),
            "conf": float(self.conf),
            "cls": self.cls,
            "color": self.color,
            "number": self.number,
            "number_conf": float(self.number_conf) if self.number_conf is not None else None,
        }


@dataclass
class VehicleDetection:
    """
    Data class for storing vehicle detection information.
    """

    img: np.ndarray
    box: np.ndarray
    conf: float
    cls: str
    color: list[int]

    def to_dict(self) -> dict:
        """
        Converts the vehicle detection data class to a dictionary.

        Returns:
            dict: The vehicle detection data class as a dictionary.
        """
        return {
            "box": self.box.astype(int).tolist(),
            "conf": float(self.conf),
            "cls": self.cls,
            "color": self.color,
        }


@dataclass
class SARKINkofaDetection:
    """
    Data class for storing SARKINkofa detection information.
    """

    vehicle: VehicleDetection | None = None
    lp: LicensePlateDetection | None = None

    def to_dict(self) -> dict:
        """
        Converts the SARKINkofa detection data class to a dictionary.

        Returns:
            dict: The SARKINkofa detection data class as a dictionary.
        """
        return {
            "vehicle": self.vehicle.to_dict() if self.vehicle is not None else None,
            "lp": self.lp.to_dict() if self.lp is not None else None,
        }
