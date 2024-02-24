from cv2.typing import MatLike

from .tools import SarkiANPD, SarkiANPR
from .types import PlateDetection, SarkiDetection, SarkiResult, VehicleDetection
from .utils import draw_bbox


class SARKINkofa:
    def __init__(self) -> None:
        # initialize the model
        self._initialize()

    def detect(
        self,
        image: MatLike,
        det_conf: float = 0.7,
        det_iou: float = 0.5,
        read_conf: float = 0.1,
        read_iou: float = 0.5,
        render: bool = False,
    ) -> SarkiDetection | None:
        result: SarkiDetection = SarkiDetection(img=image, vehicles=None)

        vehicle_out: list[VehicleDetection] | None = self._detect_vehicle(image, det_conf, det_iou)

        if vehicle_out is not None:
            for vehicle in vehicle_out:
                x1, y1, x2, y2 = vehicle.box
                plate_out: list[PlateDetection] | None = self._detect_plate(
                    image[y1:y2, x1:x2],
                    det_conf,
                    det_iou,
                    read_conf,
                    read_iou,
                )

                vehicle.plates = plate_out

        result.vehicles = vehicle_out

        if render and result.vehicles:
            result.img = self.render(image, result)

        return result

    def render(self, image: MatLike, result: SarkiDetection) -> MatLike:
        if result.vehicles is None:
            return image

        image_out: MatLike = draw_bbox(
            image.copy(),
            [v.box for v in result.vehicles],
            [v.label for v in result.vehicles],
        )

        # if there are any plate detections
        for v in result.vehicles:
            if v.plates:
                _boxes: list[tuple[int, int, int, int]] = [
                    (
                        p.box[0] + v.box[0],
                        p.box[1] + v.box[1],
                        p.box[2] + v.box[0],
                        p.box[3] + v.box[1],
                    )
                    for p in v.plates
                ]
                _labels: list[str] = [p.number if p.number else "" for p in v.plates]

                image_out = draw_bbox(image_out, _boxes, _labels)

        return image_out

    def _initialize(self) -> None:
        # load the model
        self._anpd = SarkiANPD()
        self._anpr = SarkiANPR()

    def _detect_vehicle(
        self,
        vehicle_img: MatLike,
        det_conf: float = 0.7,
        det_iou: float = 0.5,
    ) -> list[VehicleDetection] | None:
        vehicle_out: SarkiResult | None = self._anpd.detect(
            image=vehicle_img,
            conf_thresh=det_conf,
            iou_thresh=det_iou,
            exc_cls=[3],  # exclude the license plate class
            render=False,
        )

        if vehicle_out is not None:
            return [
                VehicleDetection(
                    box=box,
                    conf=conf,
                    label=label,
                )
                for box, conf, label in zip(
                    vehicle_out.boxes, vehicle_out.confs, vehicle_out.labels
                )
            ]

        return None

    def _detect_plate(
        self,
        plate_img: MatLike,
        det_conf: float = 0.7,
        det_iou: float = 0.5,
        read_conf: float = 0.1,
        read_iou: float = 0.5,
    ) -> list[PlateDetection] | None:
        plate_out: SarkiResult | None = self._anpd.detect(
            image=plate_img,
            conf_thresh=det_conf,
            iou_thresh=det_iou,
            inc_cls=[3],  # include only the license plate class
            render=False,
        )

        if plate_out is not None:
            # create a list of PlateDetection objects
            return [
                PlateDetection(
                    box=box,
                    conf=conf,
                    number=self._read_plate(
                        plate_img[box[1] : box[3], box[0] : box[2]], read_conf, read_iou
                    ),  # type: ignore
                )
                for box, conf in zip(plate_out.boxes, plate_out.confs)
            ]

        return None

    def _read_plate(
        self, number_img: MatLike, read_conf: float = 0.1, read_iou: float = 0.5
    ) -> str | None:
        number_out: SarkiResult | None = self._anpr.detect(
            image=number_img,
            conf_thresh=read_conf,
            iou_thresh=read_iou,
        )

        if number_out is not None:
            return "".join(number_out.labels)

        return None
