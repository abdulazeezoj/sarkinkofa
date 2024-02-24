import time

import cv2
import numpy as np
from cv2.typing import MatLike

from sarkinkofa import SARKINkofa
from sarkinkofa.types import SarkiDetection

print("[ INFO ] Initializing SARKINkofa...")
sarkinkofa = SARKINkofa()
print("[ INFO ] SARKINkofa initialized successfully!")

print("[ INFO ] Accessing video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[ INFO ] Video stream accessed successfully!")

fps_start_time = time.time()
fps_frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    detection: SarkiDetection | None = sarkinkofa.detect(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), render=True
    )  # type: ignore

    if detection:
        # Convert to OpenCV format
        frame: MatLike = cv2.cvtColor(np.array(detection.img), cv2.COLOR_RGB2BGR)

    # Calculate and display FPS
    fps_frame_count += 1
    if fps_frame_count % 10 == 0:
        fps: float = fps_frame_count / (time.time() - fps_start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_start_time: float = time.time()
        fps_frame_count = 0

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
