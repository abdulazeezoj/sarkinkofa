import os

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sarkinkofa import SARKINkofa
from sarkinkofa.types import SARKINkofaDetection


# Configs and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, "store")
DB_FILE = os.path.join(BASE_DIR, "db.xlsx")


def check_vehicle_in_db(detection: SARKINkofaDetection) -> pd.DataFrame | None:
    """
    Check if a vehicle is in the database.

    Args:
        vehicle (SARKINkofaDetection): The vehicle to check.

    Returns:
        pd.DataFrame | None: A DataFrame containing the vehicle information if it is in the database,
        otherwise None.
    """
    # Read database
    db = pd.read_excel(DB_FILE, sheet_name="LOG")

    # Get vehicle information
    lp_number = detection.lp.number if detection.lp is not None else None

    # Return if vehicle has no license plate
    if lp_number is None:
        return None

    # Check if vehicle is in database
    vehicle_in_db = pd.DataFrame(db.loc[db["LP_NUMBER"].str.upper() == lp_number.upper()])

    # Return if vehicle_in_db is empty
    if len(vehicle_in_db) == 0:
        return None

    return vehicle_in_db


def log_vehicle_in_db(detection: SARKINkofaDetection):
    """
    Log a vehicle in the database.

    Args:
        vehicle (SARKINkofaDetection): The vehicle to log.
    """
    # Read database
    db = pd.read_excel(DB_FILE, sheet_name="LOG")

    # Get vehicle information
    vehicle_lp_number = detection.lp.number if detection.lp is not None else None
    vehicle_lp_number_conf = detection.lp.number_conf if detection.lp is not None else None
    vehicle_lp_img = detection.lp.img if detection.lp is not None else None
    vehicle_img = detection.vehicle.img if detection.vehicle is not None else None

    # Check if vehicle is detected
    if vehicle_img is None:
        raise Exception("Vehicle not detected!")

    # Check if vehicle has license plate
    if vehicle_lp_number is None or vehicle_lp_img is None:
        raise Exception("Vehicle does not have a license plate.")

    vehicle_id = len(db) + 1
    vehicle_img_path = os.path.join(STORE_DIR, f"{vehicle_id}_vehicle.jpg")
    vehicle_lp_img_path = os.path.join(STORE_DIR, f"{vehicle_id}_lp.jpg")

    # Write vehicle and license plate images
    print("[ INFO ] Writing vehicle and license plate images...")
    cv2.imwrite(vehicle_img_path, vehicle_img)
    cv2.imwrite(vehicle_lp_img_path, vehicle_lp_img)
    print("[ INFO ] Vehicle and license plate images written.")

    # Get current date and time
    current_datetime = datetime.now()

    # Create vehicle log
    vehicle_log = {
        "CAR_ID": vehicle_id,
        "CAR_IMG": vehicle_img_path,
        "LP_IMG": vehicle_lp_img_path,
        "LP_NUMBER": vehicle_lp_number,
        "LP_CONF": vehicle_lp_number_conf,
        "ENTRY_DATE": f"{current_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
    }
    vehicle_df = pd.DataFrame([vehicle_log])

    # Log vehicle in database
    db = pd.concat([db, vehicle_df], ignore_index=True)

    # Save database, preserve other sheets
    with pd.ExcelWriter(
        DB_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        # Write database
        db.to_excel(writer, sheet_name="LOG", index=False)


def log_vehicle_exit_db(detection: SARKINkofaDetection):
    """
    Log a vehicle exit in the database.

    Args:
        vehicle (SARKINkofaDetection): The vehicle to log.
    """
    # Read database
    db = pd.read_excel(DB_FILE, sheet_name="TRAFFIC")

    # Get vehicle in database
    vehicle_in_db = check_vehicle_in_db(detection)

    # Check if vehicle is in database
    if vehicle_in_db is None:
        raise Exception("Vehicle not in database!")

    # Get vehicle ID
    vehicle_id = vehicle_in_db["CAR_ID"].values[0]

    # Get vehicle from database
    vehicle_df = db[
        (db["CAR_ID"] == vehicle_id) & ~(db["ENTRY_DATE"].isna()) & (db["EXIT_DATE"].isna())
    ].copy()

    # Sort vehicle database by entry date
    vehicle_df = vehicle_df.sort_values(by=["ENTRY_DATE"], ascending=False)

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
    vehicle_entry = vehicle_df["ENTRY_DATE"].values[0]
    vehicle_exit = vehicle_df["EXIT_DATE"].values[0]

    # Get current date and time
    current_datetime = datetime.now()

    # Calculate vehicle duration in hours
    vehicle_duration = (
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
        db.to_excel(writer, sheet_name="TRAFFIC", index=False)


def log_vehicle_entry_db(detection: SARKINkofaDetection):
    """
    Log a vehicle entry in the database.

    Args:
        vehicle (SARKINkofaDetection): The vehicle to log.
    """
    # Read database
    db = pd.read_excel(DB_FILE, sheet_name="TRAFFIC")

    # Get vehicle in database
    vehicle_in_db = check_vehicle_in_db(detection)

    # Check if vehicle is in database
    if vehicle_in_db is None:
        raise Exception("Vehicle not in database!")

    # Get vehicle ID
    vehicle_id = vehicle_in_db["CAR_ID"].values[0]

    # Get vehicle from database
    vehicle_df = db[
        (db["CAR_ID"] == vehicle_id) & ~(db["ENTRY_DATE"].isna()) & (db["EXIT_DATE"].isna())
    ].copy()

    # Sort vehicle database by entry date
    vehicle_df = vehicle_df.sort_values(by=["ENTRY_DATE"], ascending=False)

    # Check if vehicle is in database
    if not vehicle_df.empty:
        raise Exception("Vehicle entry already logged!")

    # Get current date and time
    current_datetime = datetime.now()

    # Create vehicle log
    vehicle_log = {
        "CAR_ID": vehicle_id,
        "ENTRY_DATE": f"{current_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
        "EXIT_DATE": None,
        "DURATION": None,
    }
    vehicle_df = pd.DataFrame([vehicle_log])

    # Log vehicle entry in database
    db = pd.concat([db, vehicle_df], ignore_index=True)

    # Save database
    with pd.ExcelWriter(
        DB_FILE,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        # Write database
        db.to_excel(writer, sheet_name="TRAFFIC", index=False)


def show_vehicle_detection(
    detection: SARKINkofaDetection, frame: np.ndarray, window_name: str = "SARKINkofa"
):
    """
    Show a vehicle detection on a frame.

    Args:
        vehicle (SARKINkofaDetection): The vehicle detection to show.
        frame (np.ndarray): The frame to show the vehicle detection on.
        window_name (str): The name of the window to show the vehicle detection on.
    """
    # Get vehicle bounding box
    vehicle_bbox = detection.vehicle.box if detection.vehicle is not None else None

    # Get vehicle license plate bounding box
    vehicle_lp_bbox = detection.lp.box if detection.lp is not None else None

    # Check if vehicle bounding box is not None
    if vehicle_bbox is None:
        return

    # Draw vehicle bounding box
    cv2.rectangle(
        frame,
        (vehicle_bbox[0], vehicle_bbox[1]),
        (vehicle_bbox[2], vehicle_bbox[3]),
        (0, 255, 0),
        2,
    )

    # Check if vehicle license plate bounding box is not None
    if vehicle_lp_bbox is not None:
        # Draw vehicle license plate bounding box
        cv2.rectangle(
            frame,
            (vehicle_lp_bbox[0] + vehicle_bbox[0], vehicle_lp_bbox[1] + vehicle_bbox[1]),
            (vehicle_lp_bbox[2] + vehicle_bbox[0], vehicle_lp_bbox[3] + vehicle_bbox[1]),
            (0, 0, 255),
            2,
        )

    # Show vehicle detection
    cv2.imshow(window_name, frame)


def show_message(message: str, frame: np.ndarray, window_name: str = "SARKINkofa"):
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
sarkinkofa = SARKINkofa("n")
print("[ INFO ] SARKINkofa initialized successfully!")

print("[ INFO ] Accessing video stream...")
cap = cv2.VideoCapture(0)
print("[ INFO ] Video stream accessed successfully!")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Show image
    cv2.imshow("SARKINkofa", frame)

    # Break if ESC pressed
    key = cv2.waitKey(1)

    if key == 27:  # ESC Key
        print("[ INFO ] Shutting down...")
        break

    elif key == 2:  # LEFT Arrow Key
        print("[ INFO ] Checking out exiting vehicle...")

        # TODO: Detect vehicle
        print("[ INFO ] Detecting vehicle...")
        detection = sarkinkofa.detect(frame)

        # TODO: Check if vehicle is not None
        if detection.vehicle is None:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle license plate is not None
        if detection.lp is None:
            print("[ INFO ] No vehicle license plate detected!")

            # Show message
            show_message("No vehicle license plate detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Show vehicle detection
        show_vehicle_detection(detection, frame)

        # TODO: Check if Vehicle is in database
        print(f"[ INFO ] Checking if vehicle {detection.lp.number} is in database...")
        vehicle_in_db = check_vehicle_in_db(detection)

        if vehicle_in_db is None:  # Vehicle not in database
            print(f"[ INFO ] Vehicle {detection.lp.number} not in database!")
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

            print(f"[ INFO ] Vehicle {detection.lp.number} logged in database!")

        # TODO: Log vehicle exit
        try:
            print(f"[ INFO ] Logging vehicle {detection.lp.number} exit...")
            log_vehicle_exit_db(detection)
            print(f"[ INFO ] Vehicle {detection.lp.number} exit logged!")

            # Show message
            show_message(f"Vehicle {detection.lp.number} exit logged!", frame)
        except Exception as e:
            print(f"[ ERROR ] {e}")

            # Show message
            show_message(f"Error: {e}", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

    elif key == 3:  # RIGHT Arrow Key
        print("[ INFO ] Checking out entering vehicle...")

        # TODO: Detect vehicle
        print("[ INFO ] Detecting vehicle...")
        detection = sarkinkofa.detect(frame)

        # TODO: Check if vehicle is not None
        if detection is None:
            print("[ INFO ] No vehicle detected!")

            # Show message
            show_message("No vehicle detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Check if vehicle license plate is not None
        if detection.lp is None:
            print("[ INFO ] No vehicle license plate detected!")

            # Show message
            show_message("No vehicle license plate detected!", frame)

            # Wait for 5 seconds
            cv2.waitKey(5000)

            # Continue
            continue

        # TODO: Show vehicle detection
        show_vehicle_detection(detection, frame)

        # TODO: Check if Vehicle is in database
        print(f"[ INFO ] Checking if vehicle {detection.lp.number} is in database...")
        vehicle_in_db = check_vehicle_in_db(detection)

        if vehicle_in_db is None:  # Vehicle not in database
            print(f"[ INFO ] Vehicle {detection.lp.number} not in database!")
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

            print(f"[ INFO ] Vehicle {detection.lp.number} logged in database!")

        # TODO: Log entry and entry time
        try:
            print(f"[ INFO ] Logging vehicle {detection.lp.number} entry...")
            log_vehicle_entry_db(detection)
            print(f"[ INFO ] Vehicle {detection.lp.number} entry logged!")

            # Show message
            show_message(f"Vehicle {detection.lp.number} entry logged in database!", frame)
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
