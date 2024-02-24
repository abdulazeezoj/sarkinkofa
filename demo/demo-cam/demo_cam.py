import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from pandas import DataFrame
from PIL import Image

from sarkinkofa import SARKINkofa
from sarkinkofa.types import PlateDetection, SarkiDetection, VehicleDetection

# Configs and constants
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
STORE_DIR: str = os.path.join(BASE_DIR, "store")
DB_FILE: str = os.path.join(STORE_DIR, "_store.xlsx")


def check_vehicle_in_db(detection: SarkiDetection) -> pd.DataFrame | None:
    """
    Check if a vehicle is in the database.

    Args:
        vehicle (SarkiDetection): The vehicle to check.

    Returns:
        pd.DataFrame | None: A DataFrame containing the vehicle information if it is in the database,
        otherwise None.
    """
    # Get vehicle information
    if detection.vehicles is None or len(detection.vehicles) == 0:
        return None

    if detection.vehicles[0].plates is None or len(detection.vehicles[0].plates) == 0:
        return None

    plate: PlateDetection = detection.vehicles[0].plates[0]

    # Read database
    db: DataFrame = pd.read_excel(DB_FILE, sheet_name="LOG")  # type: ignore

    # Return if vehicle has no license plate
    if plate.number is None:
        return None

    # Check if vehicle is in database
    vehicle_in_db = pd.DataFrame(db.loc[db["PLATE_NUMBER"].str.upper() == plate.number.upper()])  # type: ignore

    # Return if vehicle_in_db is empty
    if len(vehicle_in_db) == 0:
        return None

    return vehicle_in_db


def log_vehicle_in_db(detection: SarkiDetection) -> None:
    """
    Log a vehicle in the database.

    Args:
        vehicle (SarkiDetection): The vehicle to log.
    """
    # Get vehicle information
    if detection.vehicles is None or len(detection.vehicles) == 0:
        return None

    if detection.vehicles[0].plates is None or len(detection.vehicles[0].plates) == 0:
        return None

    detection_img: MatLike = detection.img
    vehicle: VehicleDetection = detection.vehicles[0]
    plate: PlateDetection = detection.vehicles[0].plates[0]

    # Read database
    db: DataFrame = pd.read_excel(DB_FILE, sheet_name="LOG")  # type: ignore

    vehicle_id: int = len(db) + 1
    vehicle_box: tuple[int, int, int, int] = vehicle.box
    plate_box: tuple[int, int, int, int] = plate.box
    vehicle_img: MatLike = detection_img[
        vehicle_box[1] : vehicle_box[3], vehicle_box[0] : vehicle_box[2]
    ]
    plate_img: MatLike = vehicle_img[plate_box[1] : plate_box[3], plate_box[0] : plate_box[2]]
    vehicle_img_path: str = os.path.join(STORE_DIR, f"{vehicle_id}_vehicle.jpg")
    plate_img_path: str = os.path.join(STORE_DIR, f"{vehicle_id}_plate.jpg")

    # Write vehicle and license plate images
    print("[ INFO ] Writing vehicle and license plate images...")
    cv2.imwrite(vehicle_img_path, cv2.cvtColor(vehicle_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(plate_img_path, cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR))
    print("[ INFO ] Vehicle and license plate images written.")

    # Get current date and time
    current_datetime: datetime = datetime.now()

    # Create vehicle log
    vehicle_log: dict[str, int | str | float | None] = {
        "VEHICLE_ID": vehicle_id,
        "VEHICLE_IMG": vehicle_img_path,
        "PLATE_IMG": plate_img_path,
        "PLATE_NUMBER": plate.number,
        "ENTRY_DATE": f"{current_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
    }
    vehicle_df = pd.DataFrame([vehicle_log])

    # Log vehicle in database
    db = pd.concat([db, vehicle_df], ignore_index=True)  # type: ignore

    # Save database, preserve other sheets
    with pd.ExcelWriter(
        DB_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        # Write database
        db.to_excel(writer, sheet_name="LOG", index=False)  # type: ignore


def log_vehicle_exit_db(detection: SarkiDetection) -> None:
    """
    Log a vehicle exit in the database.

    Args:
        vehicle (SarkiDetection): The vehicle to log.
    """
    # Read database
    db: DataFrame = pd.read_excel(DB_FILE, sheet_name="TRAFFIC")  # type: ignore

    # Get vehicle in database
    vehicle_in_db: DataFrame | None = check_vehicle_in_db(detection)

    # Check if vehicle is in database
    if vehicle_in_db is None:
        raise Exception("Vehicle not in database!")

    # Get vehicle ID
    vehicle_id: Any = vehicle_in_db["VEHICLE_ID"].values[0]  # type: ignore

    # Get vehicle from database
    vehicle_df = db[
        (db["VEHICLE_ID"] == vehicle_id) & ~(db["ENTRY_DATE"].isna()) & (db["EXIT_DATE"].isna())
    ].copy()  # type: ignore

    # Sort vehicle database by entry date
    vehicle_df: Any = vehicle_df.sort_values(by=["ENTRY_DATE"], ascending=False)

    # Check if there is a vehicle entry
    if vehicle_df.empty:
        raise Exception("Vehicle entry not logged!")

    # Check if vehicle entry is more than 1
    if len(vehicle_df) > 1:
        raise Exception("Vehicle entry logged more than once!")

    # Check if vehicle exit is already logged
    if not vehicle_df["EXIT_DATE"].isna().values[0]:
        raise Exception("Vehicle exit already logged!")

    # Get vehicle entry and entry time
    vehicle_entry: Any = vehicle_df["ENTRY_DATE"].values[0]

    # Get current date and time
    current_datetime: datetime = datetime.now()

    # Calculate vehicle duration in hours
    vehicle_duration: float = (
        current_datetime - datetime.strptime(vehicle_entry, "%Y-%m-%d %H:%M:%S")
    ).total_seconds()

    # Update vehicle log
    vehicle_df["EXIT"] = True
    vehicle_df["EXIT_DATE"] = f"{current_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    vehicle_df["DURATION"] = vehicle_duration

    # Log vehicle exit in database
    db.iloc[vehicle_df.index] = vehicle_df

    # Save database
    with pd.ExcelWriter(
        DB_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        # Write database
        db.to_excel(writer, sheet_name="TRAFFIC", index=False)  # type: ignore


def log_vehicle_entry_db(detection: SarkiDetection) -> None:
    """
    Log a vehicle entry in the database.

    Args:
        vehicle (SarkiDetection): The vehicle to log.
    """
    # Read database
    db: DataFrame = pd.read_excel(DB_FILE, sheet_name="TRAFFIC")  # type: ignore

    # Get vehicle in database
    vehicle_in_db: DataFrame | None = check_vehicle_in_db(detection)

    # Check if vehicle is in database
    if vehicle_in_db is None:
        raise Exception("Vehicle not in database!")

    # Get vehicle ID
    vehicle_id: Any = vehicle_in_db["VEHICLE_ID"].values[0]  # type: ignore

    # Get vehicle from database
    vehicle_df: Any = db[
        (db["VEHICLE_ID"] == vehicle_id) & ~(db["ENTRY_DATE"].isna()) & (db["EXIT_DATE"].isna())
    ].copy()  # type: ignore

    # Sort vehicle database by entry date
    vehicle_df = vehicle_df.sort_values(by=["ENTRY_DATE"], ascending=False)

    # Check if vehicle is in database
    if not vehicle_df.empty:
        raise Exception("Vehicle entry already logged!")

    # Get current date and time
    current_datetime: datetime = datetime.now()

    # Create vehicle log
    vehicle_log: dict[str, Any | str | None] = {
        "VEHICLE_ID": vehicle_id,
        "ENTRY_DATE": f"{current_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
        "EXIT_DATE": None,
        "DURATION": None,
    }
    vehicle_df = pd.DataFrame([vehicle_log])

    # Log vehicle entry in database
    db = pd.concat([db, vehicle_df], ignore_index=True)  # type: ignore

    # Save database
    with pd.ExcelWriter(
        DB_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        # Write database
        db.to_excel(writer, sheet_name="TRAFFIC", index=False)  # type: ignore


def show_vehicle_detection(detection: SarkiDetection, window_name: str = "SARKINkofa") -> None:
    """
    Show a vehicle detection on a frame.

    Args:
        vehicle (SarkiDetection): The vehicle detection to show.
        frame (MatLike): The frame to show the vehicle detection on.
        window_name (str): The name of the window to show the vehicle detection on.
    """
    # Show vehicle detection
    cv2.imshow(window_name, np.array(detection.img))


def show_message(message: str, frame: MatLike, window_name: str = "SARKINkofa") -> None:
    """
    Show a message on a frame.

    Args:
        message (str): The message to show.
        frame (np.ndarray): The frame to show the message on.
        window_name (str): The name of the window to show the message on.
    """

    # Get message size
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

    # Get message coordinates
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = int((frame.shape[0] + text_size[1]) / 2)

    # Show message
    cv2.putText(
        frame,
        message,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Show image
    cv2.imshow(window_name, frame)


print("[ INFO ] Initializing SARKINkofa...")
sarkinkofa = SARKINkofa()
print("[ INFO ] SARKINkofa initialized successfully!")

print("[ INFO ] Accessing video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[ INFO ] Video stream accessed successfully!")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Show image
    cv2.imshow("SARKINkofa", frame)

    # Break if ESC pressed
    key: int = cv2.waitKey(1)

    if key == 27:  # ESC Key
        print("[ INFO ] Shutting down...")
        break

    elif key == 111:  # o Key for outgoing vehicle
        print("[ INFO ] Checking out exiting vehicle...")

        # TODO: Detect vehicle
        print("[ INFO ] Detecting vehicle...")
        _frame: MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection: SarkiDetection | None = sarkinkofa.detect(Image.fromarray(_frame), render=True)  # type: ignore

        if detection is None:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle is not None
        if detection.vehicles is None or len(detection.vehicles) == 0:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle license plate is not None
        if detection.vehicles[0].plates is None or len(detection.vehicles[0].plates) == 0:
            print("[ INFO ] No vehicle license plate detected!")

            # Show message
            show_message("No vehicle license plate detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Show vehicle detection
        show_vehicle_detection(detection)

        # TODO: Check if Vehicle is in database
        print("[ INFO ] Checking if vehicle is in database...")
        vehicle_in_db = check_vehicle_in_db(detection)

        if vehicle_in_db is None:  # Vehicle not in database
            print("[ INFO ] Vehicle not in database!")
            print("[ INFO ] Creating new entry in database...")

            # TODO: Create new entry in database
            try:
                log_vehicle_in_db(detection)
            except Exception as e:
                print(f"[ ERROR ] {e}")

                # Show message
                show_message(f"Error: {e}", frame)

                # Wait for 5 seconds
                cv2.waitKey(5000)

                # Continue
                continue

            print("[ INFO ] Vehicle logged in database!")

        # TODO: Log vehicle exit
        try:
            print("[ INFO ] Logging vehicle exit...")
            log_vehicle_exit_db(detection)
            print("[ INFO ] Vehicle exit logged!")

            # Show message
            show_message("Vehicle exit logged!", frame)
        except Exception as e:
            print(f"[ ERROR ] {e}")

            # Show message
            show_message(f"Error: {e}", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

    elif key == 105:  # i Key for incoming vehicle
        print("[ INFO ] Checking out entering vehicle...")

        # TODO: Detect vehicle
        print("[ INFO ] Detecting vehicle...")
        _frame: MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection: SarkiDetection | None = sarkinkofa.detect(Image.fromarray(_frame), render=True)  # type: ignore

        if detection is None:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle is not None
        if detection.vehicles is None or len(detection.vehicles) == 0:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle license plate is not None
        if detection.vehicles[0].plates is None or len(detection.vehicles[0].plates) == 0:
            print("[ INFO ] No vehicle license plate detected!")

            # Show message
            show_message("No vehicle license plate detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Show vehicle detection
        show_vehicle_detection(detection)

        # TODO: Check if Vehicle is in database
        print("[ INFO ] Checking if vehicle is in database...")
        vehicle_in_db: DataFrame | None = check_vehicle_in_db(detection)

        if vehicle_in_db is None:  # Vehicle not in database
            print("[ INFO ] Vehicle not in database!")
            print("[ INFO ] Creating new entry in database...")

            # TODO: Create new entry in database
            try:
                log_vehicle_in_db(detection)
            except Exception as e:
                print(f"[ ERROR ] {e}")

                # Show message
                show_message(f"Error: {e}", frame)

                # Wait for 5 seconds
                cv2.waitKey(5000)

                # Continue
                continue

            print("[ INFO ] Vehicle logged in database!")

        # TODO: Log entry and entry time
        try:
            print("[ INFO ] Logging vehicle entry...")
            log_vehicle_entry_db(detection)
            print("[ INFO ] Vehicle entry logged!")

            # Show message
            show_message("Vehicle entry logged in database!", frame)
        except Exception as e:
            print(f"[ ERROR ] {e}")

            # Show message
            show_message(f"Error: {e}", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

# Release video capture
print("[ INFO ] Releasing video stream...")
cap.release()
cv2.destroyAllWindows()
print("[ INFO ] Video stream released successfully!")
print("[ INFO ] System shutdown successfully!")
