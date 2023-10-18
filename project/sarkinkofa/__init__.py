import os

import pathlib
import random
import numpy as np
from ultralytics import YOLO

from .types import LicensePlateDetection, VehicleDetection, SARKINkofaDetection
from .constants import MODEL_DIR


# Set random seed for reproducibility
random.seed(0)


class SARKINkofa:
    model_size_all = ["n", "s", "m", "l"]

    def __init__(self, model_size: str) -> None:
        """
        Initializes a SARKINkofa object.

        Args:
        size (str): The size of the SARKINkofa model to use. Must be one of "n", "s", "m", or "l".
        model_dir (str): The directory containing the SARKINkofa model.
        """
        # Check if size is valid
        if model_size not in self.model_size_all:
            raise Exception(f"Invalid model size! Must be one of {self.model_size_all}.")

        # Check if model directory exists
        if not os.path.isdir(MODEL_DIR):
            raise Exception(f"Model directory {MODEL_DIR} does not exist!")

        # Initialize model size
        self.model_size = model_size.lower()

        # Load YOLO model
        self.anpd_model = YOLO(os.path.join(MODEL_DIR, f"anpdv8{self.model_size}.pt"))
        self.anpd_model_classes = (
            pathlib.Path(os.path.join(MODEL_DIR, "anpdv8.names")).read_text().splitlines()
        )
        self.anpd_model_colors = [
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            for _ in range(len(self.anpd_model_classes))
        ]

        self.anpr_model = YOLO(os.path.join(MODEL_DIR, f"anprv8{self.model_size}.pt"))
        self.anpr_model_classes = (
            pathlib.Path(os.path.join(MODEL_DIR, "anprv8.names")).read_text().splitlines()
        )
        self.anpr_model_colors = [
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            for _ in range(len(self.anpr_model_classes))
        ]

    def __call__(self, img) -> SARKINkofaDetection:
        """
        Detects vehicles and license plates in an image.

        Args:
            img: The input image to detect vehicles and license plates in.

        Returns:
            A SARKINkofaDetection object containing the detected vehicle and license plate.
        """
        # Detect vehicle
        vehicle = self._detect_vehicle(img)

        # Check if vehicle is not None
        if vehicle is None:
            return SARKINkofaDetection()

        # Detect license plate
        lp = self._detect_lp(vehicle.img)

        # Return SARKINkofa detection
        return SARKINkofaDetection(vehicle=vehicle, lp=lp)

    def detect(self, img) -> SARKINkofaDetection:
        """
        Detects vehicles and license plates in an image.

        Args:
            img: The input image to detect vehicles and license plates in.

        Returns:
            A SARKINkofaDetection object containing the detected vehicle and license plate.
        """
        # Detect vehicle
        vehicle = self._detect_vehicle(img)

        # Check if vehicle is not None
        if vehicle is None:
            return SARKINkofaDetection()

        # Detect license plate
        lp = self._detect_lp(vehicle.img)

        # Return SARKINkofa detection
        return SARKINkofaDetection(vehicle=vehicle, lp=lp)

    def _read_lp(self, img) -> tuple[str, float] | None:
        """
        Detects license plate in an image and returns its information.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            LicensePlateDetection | None: The license plate detection result.
        """
        # Detect license plate number
        results = self.anpr_model(img, verbose=False)

        # Extract license plate number
        names = results[0].names
        boxes = results[0].boxes.data.numpy()

        # Return if no license plate is detected
        if len(names) == 0:
            return None

        # Sort boxes by x1 coordinate (left to right)
        boxes = boxes[boxes[:, 0].argsort()].tolist()

        # Add names to boxes
        labels = [names[box[-1]] for box in boxes]

        # Add labels column to boxes
        boxes = [box + [label] for box, label in zip(boxes, labels)]

        # Assemble license plate number
        lpn = "".join([box[-1] for box in boxes])
        lpn_conf = np.mean([box[4] for box in boxes], dtype=float)

        return lpn, lpn_conf

    def _detect_lp(self, img) -> LicensePlateDetection | None:
        """
        Detects license plate in an image and returns its information.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            LicensePlateDetection | None: The license plate detection result.
        """
        # Detect license plate
        anpd_results = self.anpd_model(img, classes=[3], verbose=False)

        # Extract license plate
        boxes = anpd_results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = anpd_results[0].boxes.conf.cpu().numpy()
        cls_ids = anpd_results[0].boxes.cls.cpu().numpy().astype(int)
        cls_labels = [self.anpd_model_classes[cls_id] for cls_id in cls_ids]
        cls_colors = [self.anpd_model_colors[cls_id] for cls_id in cls_ids]

        # Return if no license plate is detected
        if len(boxes) == 0:
            return None

        # Get license plate with highest confidence
        lp_idx = confs.argmax()
        lp_box = boxes[lp_idx]
        lp_conf = confs[lp_idx]
        lp_cls = cls_labels[lp_idx]
        lp_color = cls_colors[lp_idx]
        lp_img = img[lp_box[1] : lp_box[3], lp_box[0] : lp_box[2]]

        # Detect license plate number
        anpr_result = self._read_lp(lp_img)

        # Extract license plate number
        if not anpr_result:
            lp_number, lp_number_conf = None, None
        else:
            lp_number, lp_number_conf = anpr_result

        # Return license plate
        return LicensePlateDetection(
            img=lp_img,
            box=lp_box,
            conf=lp_conf,
            cls=lp_cls,
            color=lp_color,
            number=lp_number,
            number_conf=lp_number_conf,
        )

    def _detect_vehicle(self, img) -> VehicleDetection | None:
        # Detect vehicle
        anpd_results = self.anpd_model(img, classes=[1], verbose=False)

        # Extract vehicle information
        boxes = anpd_results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = anpd_results[0].boxes.conf.cpu().numpy()
        cls_ids = anpd_results[0].boxes.cls.cpu().numpy().astype(int)
        cls_labels = [self.anpd_model_classes[cls_id] for cls_id in cls_ids]
        cls_colors = [self.anpd_model_colors[cls_id] for cls_id in cls_ids]

        # Return if no vehicle is detected
        if len(boxes) == 0:
            return None

        # Get vehicle with highest confidence
        vehicle_idx = confs.argmax()
        vehicle_box = boxes[vehicle_idx]
        vehicle_conf = confs[vehicle_idx]
        vehicle_cls = cls_labels[vehicle_idx]
        vehicle_color = cls_colors[vehicle_idx]
        vehicle_img = img[vehicle_box[1] : vehicle_box[3], vehicle_box[0] : vehicle_box[2]]

        # Detect license plate
        vehicle_lp = self._detect_lp(vehicle_img)

        # Return vehicle
        return VehicleDetection(
            img=vehicle_img,
            box=vehicle_box,
            conf=vehicle_conf,
            cls=vehicle_cls,
            color=vehicle_color,
        )
