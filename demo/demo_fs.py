import os
from sarkinkofa.tools import SARKINKofaFSWatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


if __name__ == "__main__":
    # Initialize watcher
    print("[ INFO ] Initializing watcher...")
    image_watcher = SARKINKofaFSWatcher(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)
    print("[ INFO ] Watcher initialized!")

    try:
        # Start watching
        print("[ INFO ] Starting watcher...")
        image_watcher.start_watching()
    except KeyboardInterrupt:
        print("[ INFO ] Stopping watcher...")
        image_watcher.stop_watching()
        print("[ INFO ] Watcher stopped!")
