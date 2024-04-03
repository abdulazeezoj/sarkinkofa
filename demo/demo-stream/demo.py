import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
from cv2.typing import MatLike

from sarkinkofa import SARKINkofa
from sarkinkofa.types import SarkiDetection


class VideoStream:
    def __init__(self, path: str | int, queue_size: int = 128) -> None:
        try:
            # Initialize the video stream and read the first frame
            self.stream = cv2.VideoCapture(path)

            # Initialize the queue used to store frames read from the video stream
            self.Q: Queue[MatLike] = Queue(maxsize=queue_size)

            # Initialize the variable used to indicate if the thread should be stopped
            self.stopped = False
        except Exception as e:
            print(f"[ ERROR ] {e}")

    def start(self) -> None:
        # Start a thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self) -> None:
        # Keep looping infinitely
        while True:
            # If the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # Otherwise, ensure the queue has room in it
            if not self.Q.full():
                # Read the next frame from the stream
                _stream: tuple[bool, MatLike] = self.stream.read()
                grabbed: bool = _stream[0]
                frame: MatLike = _stream[1]

                # If the `grabbed` boolean is `False`, then we have reached the end of the video
                if not grabbed:
                    self.stop()
                    return

                # Add the frame to the queue
                self.Q.put(frame)

    def read(self) -> MatLike:
        # Return next frame in the queue
        return self.Q.get()

    def more(self) -> bool:
        # Return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self) -> None:
        # Indicate that the thread should be stopped
        self.stopped = True


print("[ INFO ] Initializing SARKINkofa...")
sarkinkofa = SARKINkofa()
print("[ INFO ] SARKINkofa initialized successfully!")

print("[ INFO ] Accessing video stream...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[ INFO ] Video stream accessed successfully!")

prev_frame_time: float = 0
new_frame_time: float = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    detection: SarkiDetection | None = sarkinkofa.detect(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), render=True
    )

    if detection:
        # Convert to OpenCV format
        frame: MatLike = cv2.cvtColor(np.array(detection.img), cv2.COLOR_RGB2BGR)

    # Calculate and display FPS
    new_frame_time: float = time.time()
    fps: int = (
        int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time != new_frame_time else 0
    )
    prev_frame_time: float = new_frame_time

    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("SARKINkofa", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[ INFO ] Shutting down...")
        break


# Release video capture
print("[ INFO ] Releasing video stream...")
cap.release()
cv2.destroyAllWindows()
print("[ INFO ] Video stream released successfully!")
print("[ INFO ] System shutdown successfully!")
