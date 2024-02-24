import time

import cv2
import typer
from cv2.typing import MatLike

from sarkinkofa import SARKINkofa
from sarkinkofa.types import SarkiDetection

print("[ INFO ] Initializing SARKINkofa...")
sarkinkofa = SARKINkofa()
print("[ INFO ] SARKINkofa initialized successfully!")

app = typer.Typer()


@app.command()
def fs(
    input_folder: str = typer.Argument(..., help="Folder to watch."),
    output_folder: str = typer.Argument(..., help="Folder to store output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode."),
) -> None:
    """
    SARKINKofa File System Watcher CLI.
    """
    typer.echo("[ INFO ] Not implemented yet!")


@app.command()
def cam(
    stream: str = typer.Argument(
        ..., help="Video stream to watch. Can be a file path, URL, or camera index."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode."),
) -> None:
    """
    SARKINKofa Camera/Video Stream Interface CLI.
    """
    typer.echo("[ INFO ] Accessing video stream...")
    cap = cv2.VideoCapture(int(stream) if stream.isdigit() else stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    typer.echo("[ INFO ] Video stream accessed successfully!")

    # Initialize FPS counter
    fps_start_time = 0
    fps_frame_count = 0

    # Start streaming
    while True:
        success, frame = cap.read()

        if not success:
            break

        detection: SarkiDetection | None = sarkinkofa.detect(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), render=True
        )

        if detection:
            # Convert to OpenCV format
            frame: MatLike = cv2.cvtColor(detection.img, cv2.COLOR_RGB2BGR)

        # Calculate and display FPS
        fps_frame_count += 1
        if fps_frame_count % 10 == 0:
            fps: float = fps_frame_count / (time.time() - fps_start_time)
            cv2.putText(
                frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            fps_start_time: float = time.time()
            fps_frame_count = 0

        # Display frame
        cv2.imshow("SARKINkofa", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            typer.echo("[ INFO ] Shutting down...")
            break

    # Release video capture
    typer.echo("[ INFO ] Releasing video stream...")
    cap.release()
    cv2.destroyAllWindows()
    typer.echo("[ INFO ] Video stream released successfully!")


if __name__ == "__main__":
    app()
