import os
from datetime import datetime
from typing import List, Literal, Tuple

from sqlalchemy import DateTime, Engine, ForeignKey, Select, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileSystemEventHandler

from sarkinkofa.tools import SarkiFSWatcher

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
INC_DIR: str = os.path.join(BASE_DIR, "inc")
OUT_DIR: str = os.path.join(BASE_DIR, "out")
LOG_DIR: str = os.path.join(BASE_DIR, "log")

# Create input and output directories if they don't exist
if not os.path.exists(INC_DIR):
    os.makedirs(INC_DIR)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    __tablename__: str = "vehicle"

    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(20), unique=True)
    vehicle_image: Mapped[str] = mapped_column(String)
    plate_image: Mapped[str] = mapped_column(String)
    number_file: Mapped[str] = mapped_column(String)

    traffics: Mapped[List["Traffic"]] = relationship(
        back_populates="vehicle", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Vehicle(number={self.number})>"


class Traffic(Base):
    __tablename__: str = "traffic"

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicle.id"), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    entry_image: Mapped[str] = mapped_column(String)
    exit_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    exit_image: Mapped[str] = mapped_column(String, nullable=True)

    vehicle: Mapped["Vehicle"] = relationship(back_populates="traffics")

    def __repr__(self) -> str:
        return f"<Traffic(vehicle={self.vehicle.number}, entry_time={self.entry_time}, exit_time={self.exit_time})>"


class SarkiFSHandler(FileSystemEventHandler):
    def __init__(self, db_path: str = "sqlite:///_db.sqlite3") -> None:
        super().__init__()

        self.engine: Engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.files: dict[str, dict[str, str | None]] = {}

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        _src_path: str = event.src_path

        if not event.is_directory:
            if _src_path.lower().endswith((".png", ".jpg", ".txt")):
                if INC_DIR in _src_path:
                    print(f"[ INFO ] New File in 'in' directory: {_src_path}")
                    _f_id: str = self._parse(_src_path)

                    self._process(_f_id, type="inc")
                elif OUT_DIR in _src_path:
                    print(f"[ INFO ] New File in 'out' directory: {_src_path}")
                    _f_id: str = self._parse(_src_path)

                    self._process(_f_id, type="out")

    def _parse(self, _src_path: str) -> str:
        _f_name: str = os.path.basename(_src_path)
        _f_id: str = _f_name.split("_")[0]
        _f_type: str = _f_name.split("_")[1].split(".")[0]

        if _f_id in self.files:
            self.files[_f_id][_f_type] = _src_path
        else:
            self.files[_f_id] = {"number": None, "plate": None, "vehicle": None}
            self.files[_f_id][_f_type] = _src_path

        return _f_id

    def _process(self, id: str, type: Literal["inc", "out"]) -> None:
        # check if all files are available
        if not all(self.files[id].values()):
            print(f"[ INFO ] Waiting for other files for {id}...")
            return

        number_file: str | None = self.files[id]["number"]
        plate_file: str | None = self.files[id]["plate"]
        vehicle_file: str | None = self.files[id]["vehicle"]

        if not (number_file and plate_file and vehicle_file):
            print(f"[ INFO ] Waiting for other files for {id}...")
            return

        print("[ INFO ] Processing all files: ")
        print(f"[ INFO ] Number file: {number_file}")
        print(f"[ INFO ] Plate file: {plate_file}")
        print(f"[ INFO ] Vehicle file: {vehicle_file}")

        if type == "inc":
            self._inc_process(number_file, plate_file, vehicle_file)

        elif type == "out":
            self._out_process(number_file, plate_file, vehicle_file)

        self.files.pop(id)

    def _inc_process(self, number_file: str, plate_file: str, vehicle_file: str) -> None:
        # process files
        number: str = self._read_file(number_file)
        vehicle_image: str = vehicle_file
        plate_image: str = plate_file

        # get vehicle
        vehicle: Vehicle | None = self._get_vehicle(number)

        if vehicle:
            print(f"[ INFO ] Vehicle already exists: {number}")

            # get traffic
            traffic: Traffic | None = self._get_traffic(vehicle)

            if traffic:
                print(f"[ WARN ] Vehicle already in traffic: {number}")

                return

            # create traffic
            entry_time: datetime = datetime.now()
            entry_image: str = vehicle_image
            self._create_traffic(vehicle, entry_time, entry_image)
            print(f"[ INFO ] Vehicle entry logged: {number}")

        else:
            print(f"[ INFO ] Creating new vehicle: {number}")
            vehicle = self._create_vehicle(number, vehicle_image, plate_image, number_file)

            # create traffic
            entry_time: datetime = datetime.now()
            entry_image: str = vehicle_image
            self._create_traffic(vehicle, entry_time, entry_image)
            print(f"[ INFO ] New vehicle created: {number}")

    def _out_process(self, number_file: str, plate_file: str, vehicle_file: str) -> None:
        # process files
        number: str = self._read_file(number_file)
        vehicle_image: str = vehicle_file
        plate_image: str = plate_file

        # get vehicle
        vehicle: Vehicle | None = self._get_vehicle(number)

        if vehicle:
            # get traffic
            traffic: Traffic | None = self._get_traffic(vehicle)

            if not traffic:
                print(f"[ WARN ] Vehicle not in traffic: {number}")

                return

            if not traffic.entry_time:
                print(f"[ WARN ] Vehicle entry not logged: {number}")

                return

            if traffic.exit_time:
                print(f"[ WARN ] Vehicle already exited: {number}")

                return

            print(f"[ INFO ] Updating traffic: {number}")
            exit_time: datetime = datetime.now()
            exit_image: str = vehicle_image
            self._update_traffic(traffic, exit_time, exit_image)
            print(f"[ INFO ] Vehicle exit logged: {number}")

        else:
            print(f"[ WARN ] Vehicle does not exist: {number}")
            vehicle = self._create_vehicle(number, vehicle_image, plate_image, number_file)

            # get traffic
            traffic: Traffic | None = self._get_traffic(vehicle)

            if not traffic:
                print(f"[ WARN ] Vehicle not in traffic: {number}")

                return

            if not traffic.entry_time:
                print(f"[ WARN ] Vehicle entry not logged: {number}")

                return

            if traffic.exit_time:
                print(f"[ WARN ] Vehicle already exited: {number}")

                return

    def _read_file(self, file: str) -> str:
        with open(file, "r") as f:
            return f.read()

    def _create_vehicle(
        self, number: str, vehicle_image: str, plate_image: str, number_file: str
    ) -> Vehicle:
        with Session(self.engine) as session:
            vehicle: Vehicle = Vehicle(
                number=number,
                vehicle_image=vehicle_image,
                plate_image=plate_image,
                number_file=number_file,
            )

            session.add(vehicle)
            session.commit()

        return vehicle

    def _get_vehicle(self, number: str) -> Vehicle | None:
        with Session(self.engine) as session:
            query: Select[Tuple[Vehicle]] = select(Vehicle).where(Vehicle.number == number)
            vehicle: Vehicle | None = session.scalars(query).first()

            return vehicle

    def _create_traffic(self, vehicle: Vehicle, entry_time: datetime, entry_image: str) -> Traffic:
        with Session(self.engine) as session:
            session.add(vehicle)

            traffic: Traffic = Traffic(
                vehicle_id=vehicle.id,
                entry_time=entry_time,
                entry_image=entry_image,
            )

            session.add(traffic)
            session.commit()

        return traffic

    def _update_traffic(self, traffic: Traffic, exit_time: datetime, exit_image: str) -> None:
        with Session(self.engine) as session:
            session.add(traffic)

            traffic.exit_time = exit_time
            traffic.exit_image = exit_image

            session.commit()

    def _get_traffic(self, vehicle: Vehicle) -> Traffic | None:
        with Session(self.engine) as session:
            session.add(vehicle)

            query: Select[Tuple[Traffic]] = (
                select(Traffic)
                .where(Traffic.vehicle_id == vehicle.id)
                .order_by(Traffic.entry_time.desc())
            )
            traffic: Traffic | None = session.scalars(query).first()

            return traffic


if __name__ == "__main__":
    # Initialize watcher
    print("[ INFO ] Initializing watcher...")
    watcher = SarkiFSWatcher(
        input_folder=[INC_DIR, OUT_DIR],
        ev_handler=SarkiFSHandler(),
        frequency=0.5,
        recursive=False,
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
