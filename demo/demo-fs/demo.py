import argparse
import os
from datetime import datetime
from typing import List, Literal, Tuple

from sqlalchemy import DateTime, Engine, ForeignKey, Select, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from sarkinkofa.tools import SarkiFSWatcher


class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    __tablename__: str = "vehicle"

    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(20), unique=True)
    vehicle_image: Mapped[str] = mapped_column(String)
    plate_image: Mapped[str] = mapped_column(String)

    traffics: Mapped[List["Traffic"]] = relationship(
        back_populates="vehicle", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Vehicle(number={self.number})>"

    def to_dict(self) -> dict[str, str]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Traffic(Base):
    __tablename__: str = "traffic"

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicle.id"), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    entry_vehicle_image: Mapped[str] = mapped_column(String)
    entry_plate_image: Mapped[str] = mapped_column(String)
    exit_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    exit_vehicle_image: Mapped[str] = mapped_column(String, nullable=True)
    exit_plate_image: Mapped[str] = mapped_column(String, nullable=True)

    vehicle: Mapped["Vehicle"] = relationship(back_populates="traffics")

    def __repr__(self) -> str:
        return f"<Traffic(vehicle={self.vehicle.number}, entry_time={self.entry_time}, exit_time={self.exit_time})>"


class SarkiFSHandler(FileSystemEventHandler):
    def __init__(self, inc_dir: str, out_dir: str, db_path: str = "sqlite:///_demo.db") -> None:
        super().__init__()

        self.engine: Engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.store: dict[str, dict[str, str | None]] = {}
        self.inc_dir: str = inc_dir
        self.out_dir: str = out_dir

    def on_created(self, event: FileSystemEvent) -> None:
        _src_path: str = event.src_path

        if not event.is_directory:
            if _src_path.lower().endswith((".jpg")):
                _f_id: str | None = self._parse(_src_path)

                if not _f_id:
                    return

                # check if source path is in input or output directory
                if self.inc_dir in _src_path:  # incoming vehicle
                    self._process(_f_id, type="inc")
                elif self.out_dir in _src_path:  # outgoing vehicle
                    self._process(_f_id, type="out")

    def _parse(self, _src_path: str) -> str | None:
        """
        Parses the source path and updates the logs with the file information.

        Args:
            _src_path (str): The source path of the file.

        Returns:
            str | None: The parsed file ID if it is not "no plate", otherwise None.
        """
        # parse file name
        _f_name: str = os.path.basename(_src_path)
        _f_split: list[str] = _f_name.split("_")
        _f_id: str = _f_split[1]
        _f_type: str = "plate" if _f_split[-1].split(".")[0] == "plate" else "vehicle"

        # return None if no plate
        if _f_id.lower() == "no plate":
            return None

        # create / update logs
        if _f_id in self.store:
            self.store[_f_id][_f_type] = _src_path
        else:
            self.store[_f_id] = {"plate": None, "vehicle": None}
            self.store[_f_id][_f_type] = _src_path

        return _f_id

    def _process(self, number: str, type: Literal["inc", "out"]) -> None:
        """
        Process the given number based on the type.

        Args:
            number (str): The number to be processed.
            type (Literal["inc", "out"]): The type of processing to be performed.

        Returns:
            None
        """
        # check if all files are available
        if not all(self.store[number].values()):
            return

        plate_file: str | None = self.store[number]["plate"]
        vehicle_file: str | None = self.store[number]["vehicle"]

        # return if any file is missing
        if not (plate_file and vehicle_file):
            return

        if type == "inc":
            self._inc_process(number, plate_file, vehicle_file)

        elif type == "out":
            self._out_process(number, plate_file, vehicle_file)

        self.store.pop(number)

    def _inc_process(self, number: str, plate_file: str, vehicle_file: str) -> None:
        # process files
        vehicle_image: str = vehicle_file
        plate_image: str = plate_file

        # get vehicle, create if not exists
        vehicle: Vehicle = self._get_create_vehicle(number, vehicle_image, plate_image)

        # log entry
        self._log_entry(
            vehicle,
            entry_time=datetime.utcnow(),
            vehicle_image=vehicle_image,
            plate_image=plate_image,
        )

    def _out_process(self, number: str, plate_file: str, vehicle_file: str) -> None:
        # process files
        vehicle_image: str = vehicle_file
        plate_image: str = plate_file

        # get vehicle, create if not exists
        vehicle: Vehicle = self._get_create_vehicle(number, vehicle_image, plate_image)

        # log exit
        self._log_exit(
            vehicle,
            exit_time=datetime.utcnow(),
            vehicle_image=vehicle_image,
            plate_image=plate_image,
        )

    def _create_vehicle(self, number: str, vehicle_image: str, plate_image: str) -> Vehicle:
        with Session(self.engine) as session:
            vehicle: Vehicle = Vehicle(
                number=number, vehicle_image=vehicle_image, plate_image=plate_image
            )

            session.add(vehicle)
            session.commit()
            session.expunge(vehicle)

        return vehicle

    def _get_vehicle(self, number: str) -> Vehicle | None:
        with Session(self.engine) as session:
            query: Select[Tuple[Vehicle]] = select(Vehicle).where(Vehicle.number == number)
            vehicle: Vehicle | None = session.scalars(query).first()

            return vehicle

    def _get_create_vehicle(self, number: str, vehicle_image: str, plate_image: str) -> Vehicle:
        _vehicle: Vehicle | None = self._get_vehicle(number)

        if not _vehicle:
            _vehicle = self._create_vehicle(number, vehicle_image, plate_image)
            self._log("Vehicle: {number} registered successfully ")

        return _vehicle

    def _create_traffic(
        self, vehicle: Vehicle, entry_time: datetime, vehicle_image: str, plate_image: str
    ) -> Traffic:
        with Session(self.engine) as session:
            traffic: Traffic = Traffic(
                vehicle_id=vehicle.id,
                entry_time=entry_time,
                entry_vehicle_image=vehicle_image,
                entry_plate_image=plate_image,
            )

            session.add(traffic)
            session.commit()
            session.expunge(traffic)

        return traffic

    def _update_traffic(
        self, traffic: Traffic, exit_time: datetime, vehicle_image: str, plate_image: str
    ) -> Traffic:
        with Session(self.engine) as session:
            session.add(traffic)

            traffic.exit_time = exit_time
            traffic.exit_vehicle_image = vehicle_image
            traffic.exit_plate_image = plate_image

            session.commit()
            session.expunge(traffic)

            return traffic

    def _get_traffic(self, vehicle: Vehicle) -> Traffic | None:
        with Session(self.engine) as session:
            query: Select[Tuple[Traffic]] = (
                select(Traffic)
                .where(Traffic.vehicle_id == vehicle.id)
                .order_by(Traffic.entry_time.desc())
            )
            traffic: Traffic | None = session.scalars(query).first()

            return traffic

    def _log_entry(
        self,
        vehicle: Vehicle,
        entry_time: datetime,
        vehicle_image: str,
        plate_image: str,
    ) -> Traffic | None:
        traffic: Traffic | None = self._get_traffic(vehicle)

        if not traffic or (traffic.entry_time and traffic.exit_time):
            traffic = self._create_traffic(
                vehicle, entry_time, vehicle_image=vehicle_image, plate_image=plate_image
            )
            self._log(f"Vehicle: {vehicle.number} ENTRY logged successfully ")

            # send notification to gate controller to open gate
            self._notify_gate("in", "open")

            return traffic

        if (traffic.entry_vehicle_image == vehicle_image) or (
            traffic.entry_plate_image == plate_image
        ):
            return traffic

        if traffic.entry_time and not traffic.exit_time:
            self._log(f"Vehicle: {vehicle.number} ENTRY logged but no EXIT", level="warn")

            # [ TODO ] send notification to security to check vehicle
            self._notify_security("check")

            return traffic

    def _log_exit(
        self,
        vehicle: Vehicle,
        exit_time: datetime,
        vehicle_image: str,
        plate_image: str,
    ) -> Traffic | None:
        traffic: Traffic | None = self._get_traffic(vehicle)

        if not traffic:
            self._log(f"Vehicle: {vehicle.number} ENTRY not found", level="warn")

            # send notification to security to check vehicle
            self._notify_security("check")

            return traffic

        if (traffic.entry_vehicle_image == vehicle_image) or (
            traffic.entry_plate_image == plate_image
        ):
            return traffic

        if traffic.entry_time and traffic.exit_time:
            self._log(f"Vehicle: {vehicle.number} already EXIT", level="warn")

            # send notification to security to check vehicle
            self._notify_security("check")

            return traffic

        if traffic.entry_time and not traffic.exit_time:
            self._update_traffic(traffic, exit_time, vehicle_image, plate_image)
            self._log(f"Vehicle: {vehicle.number} EXIT logged successfully ")

            # send notification to gate controller to open gate
            self._notify_gate("out", "open")

            return traffic

    def _notify_gate(self, gate: str, action: Literal["open", "close"]) -> None:
        # check if action is valid
        if action not in ["open", "close"]:
            self._log(f"Action: {action} not found", level="warn")
            return

        if gate == "in":
            self._log(f"Gate: {gate.upper()} - {action.upper()} requested")
        elif gate == "out":
            self._log(f"Gate: {gate.upper()} - {action.upper()} requested")
        else:
            self._log(f"Gate: {gate.upper()} not found", level="warn")

    def _notify_security(self, action: Literal["check", "action"]) -> None:
        if action == "check":
            self._log(f"Security: {action.upper()} requested")
        elif action == "action":
            self._log(f"Security: {action.upper()} requested")
        else:
            self._log(f"Security: {action} not found", level="warn")

    def _log(self, message: str, level: Literal["info", "warn", "error"] = "info") -> None:
        print(f"[ {level.upper()} ] {message}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SarkiFSWatcher")
    parser.add_argument("--inc", type=str, help="Path to the input directory")
    parser.add_argument("--out", type=str, help="Path to the output directory")
    args = parser.parse_args()

    # [ COMMAND ] python ./demo_db.py --inc <path> --out <path>

    # Create input and output directories if they don't exist
    os.makedirs(args.inc, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # Initialize watcher
    print("[ INFO ] Initializing watcher...")
    watcher = SarkiFSWatcher(
        input_folder=[args.inc, args.out],
        ev_handler=SarkiFSHandler(args.inc, args.out),
        frequency=0.5,
        recursive=True,
    )
    print("[ INFO ] Watcher initialized!")

    try:
        # Start watching
        print("[ INFO ] Starting watcher...")
        watcher.start_watching()
    except KeyboardInterrupt:
        print("[ INFO ] Stopping watcher...")
        watcher.stop_watching()
        print("[ INFO ] Watcher stopped!")
