import os

import cv2
from cv2.typing import MatLike
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileSystemEventHandler

from sarkinkofa import SARKINkofa
from sarkinkofa.tools import SarkiFSWatcher
from sarkinkofa.types import SarkiDetection

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR: str = os.path.join(BASE_DIR, "input")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")

# Create input and output directories if they don't exist
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class SarkiFSHandler(FileSystemEventHandler):
    """
    A class that handles file system events and processes images.

    Attributes:
        output_folder (str): The path to the output folder.
        verbose (bool): Whether to print verbose output.
        detector (SARKINkofa): An instance of the SARKINkofa class.
    """

    def __init__(self, output_folder: str, detector: SARKINkofa, verbose: bool = False) -> None:
        """
        Initializes an instance of SARKINKofaFSHandler.

        Args:
            output_folder (str): The path to the output folder.
            detector (SARKINkofa): An instance of the SARKINkofa class.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.output_folder: str = output_folder
        self.verbose: bool = verbose
        self.detector: SARKINkofa = detector

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        if event.is_directory:
            return

        _src_path: str = event.src_path
        if _src_path.lower().endswith((".png", ".jpg", ".jpeg")):
            if self.verbose:
                print(f"[ INFO ] New image detected: {_src_path}")
                print(f"[ INFO ] Processing image: {_src_path}")

            self._process(_src_path)

            if self.verbose:
                print(f"[ INFO ] Image processed: {_src_path}")

    def _read(self, image_path: str) -> MatLike:
        # Read image
        image: MatLike = cv2.imread(image_path)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _process(self, image_path: str) -> None:
        # Read image
        image: MatLike = self._read(image_path)

        # Detect vehicle & plate number
        detection: SarkiDetection | None = self.detector.detect(image, render=True)

        if detection:
            # Prepare output file name
            file_name: str = os.path.splitext(os.path.basename(image_path))[0]

            # Save image
            _image_path: str = os.path.join(self.output_folder, f"{file_name}.jpg")
            cv2.imwrite(_image_path, cv2.cvtColor(detection.img, cv2.COLOR_RGB2BGR))

            # Save detection
            _detection_path: str = os.path.join(self.output_folder, f"{file_name}.json")
            with open(_detection_path, "w") as f:
                f.write(detection.model_dump_json(indent=4, exclude={"img": True}))

            if self.verbose:
                print(f"[ INFO ] Image saved: {_image_path}")
                print(f"[ INFO ] Detection saved: {_detection_path}")


if __name__ == "__main__":
    # Initialize watcher
    print("[ INFO ] Initializing watcher...")
    watcher = SarkiFSWatcher(
        input_folder=INPUT_DIR,
        ev_handler=SarkiFSHandler(output_folder=OUTPUT_DIR, detector=SARKINkofa(), verbose=True),
        frequency=0.5,
        recursive=False,
    )
    print("[ INFO ] Watcher initialized!")

    try:
        # Start watching
        print("[ INFO ] Starting watcher...")
        watcher.start_watching()
        print("[ INFO ] Watcher started!")
    except KeyboardInterrupt:
        print("[ INFO ] Stopping watcher...")
        watcher.stop_watching()
        print("[ INFO ] Watcher stopped!")
