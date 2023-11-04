import os
import time
import cv2
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from . import SARKINkofa
from .types import SARKINkofaDetection


class SARKINKofaFSWatcher:
    """
    A class for watching a folder for changes and triggering an event handler.

    Attributes:
        input_folder (str): The path to the input folder.
        output_folder (str): The path to the output folder.
        observer (Observer): The observer object used to watch the input folder.
        detector (SARKINkofa): The detector object used to detect changes in the input folder.
        event_handler (SARKINKofaFSHandler): The event handler object used to handle changes in the input folder.
        verbose (bool): Whether to print verbose output.
    """

    def __init__(self, input_folder: str, output_folder: str, verbose=False):
        """
        Initializes an instance of SARKINKofaFSWatcher.

        Args:
            input_folder (str): The path to the input folder.
            output_folder (str): The path to the output folder.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.observer = Observer()
        self.detector = SARKINkofa("n")
        self.event_handler = SARKINKofaFSHandler(
            self.output_folder, self.detector, verbose=verbose
        )
        self.verbose = verbose

    def start_watching(self):
        """
        Starts watching the input folder for any changes and triggers the event handler
        """
        self.observer.schedule(self.event_handler, path=self.input_folder, recursive=False)
        self.observer.start()

        while True:
            time.sleep(1)

    def stop_watching(self):
        """
        Stops the observer and waits for it to finish.
        """
        self.observer.stop()
        self.observer.join()


class SARKINKofaFSHandler(FileSystemEventHandler):
    """
    A class that handles file system events and processes images.

    Attributes:
        output_folder (str): The path to the output folder.
        verbose (bool): Whether to print verbose output.
        detector (SARKINkofa): An instance of the SARKINkofa class.
    """

    def __init__(self, output_folder: str, detector: SARKINkofa, verbose=False):
        """
        Initializes an instance of SARKINKofaFSHandler.

        Args:
            output_folder (str): The path to the output folder.
            detector (SARKINkofa): An instance of the SARKINkofa class.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.output_folder = output_folder
        self.verbose = verbose
        self.detector = detector

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            if self.verbose:
                print(f"[ INFO ] New image detected: {event.src_path}")
                print(f"[ INFO ] Processing image: {event.src_path}")

            self.process_image(event.src_path)

            if self.verbose:
                print(f"[ INFO ] Image processed: {event.src_path}")

    def read_image(self, image_path):
        # Read image
        image = cv2.imread(image_path)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def render_image(self, image, detection: SARKINkofaDetection):
        # Convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get vehicle and plate number bounding boxes
        vehicle_bbox = detection.vehicle.box if detection.vehicle is not None else None
        plate_bbox = detection.lp.box if detection.lp is not None else None

        # Check if vehicle is detected
        if vehicle_bbox is not None:
            # Draw vehicle bounding box
            image = cv2.rectangle(
                image,
                (vehicle_bbox[0], vehicle_bbox[1]),
                (vehicle_bbox[2], vehicle_bbox[3]),
                (0, 255, 0),
                2,
            )

            # Check if plate number is detected
            if plate_bbox is not None:
                # Draw plate number bounding box
                image = cv2.rectangle(
                    image,
                    (plate_bbox[0] + vehicle_bbox[0], plate_bbox[1] + vehicle_bbox[1]),
                    (plate_bbox[2] + vehicle_bbox[0], plate_bbox[3] + vehicle_bbox[1]),
                    (0, 0, 255),
                    3,
                )

                # Get plate number text
                plate_text = detection.lp.number if detection.lp is not None else None

                # Check if plate number text is present
                if plate_text is not None:
                    # Draw plate number text
                    image = cv2.putText(
                        image,
                        plate_text,
                        (plate_bbox[0] + vehicle_bbox[0], plate_bbox[1] + vehicle_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3,
                    )

        return image

    def process_image(self, image_path):
        # Read image
        image = self.read_image(image_path)

        # Detect vehicle & plate number
        detection: SARKINkofaDetection = self.detector(image)

        # Render image
        _image = self.render_image(image, detection)

        # Prepare detection in json format
        _detection = detection.to_dict()

        # Prepare output file name
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save image
        _image_path = os.path.join(self.output_folder, f"{file_name}.jpg")
        cv2.imwrite(_image_path, _image)

        # Save detection
        _detection_path = os.path.join(self.output_folder, f"{file_name}.json")
        with open(_detection_path, "w") as f:
            json.dump(_detection, f)

        if self.verbose:
            print(f"[ INFO ] Image saved: {_image_path}")
            print(f"[ INFO ] Detection saved: {_detection_path}")
            print(f"[ INFO ] Image processed: {image_path}")
