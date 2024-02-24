from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


def xywh2xyxy(xywh: NDArray[np.intp]) -> NDArray[np.intp]:
    # make a copy of the bounding boxes
    xyxy: NDArray[np.intp] = np.copy(xywh)

    # convert the bounding boxes from [x, y, w, h] to [x1, y1, x2, y2]
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    return xyxy


def xyxy2xywh(xyxy: NDArray[np.intp]) -> NDArray[np.intp]:
    # make a copy of the bounding boxes
    xywh: NDArray[np.intp] = np.copy(xyxy)

    # convert the bounding boxes from [x1, y1, x2, y2] to [x, y, w, h]
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    return xywh


def resize(
    image: MatLike,
    size: tuple[int, int],
    aspect_ratio: bool = False,
    fill: tuple[int, int, int] | None = None,
) -> tuple[MatLike, tuple[int, int], tuple[int, int, int, int]]:
    _image: MatLike = image.copy()

    # resize the image while keeping the aspect ratio
    if aspect_ratio or fill:
        # get size of the image
        height, width, _ = _image.shape

        # calculate the aspect ratio of the image
        aspect: float = width / height

        # calculate the new size of the image
        _width: int = size[0]
        _height: int = int(_width / aspect)

        if _height >= size[1]:
            _height = size[1]
            _width = int(_height * aspect)

        # resize the image
        _image = cv2.resize(_image, (_width, _height), interpolation=cv2.INTER_LANCZOS4)

        # create a new image with the desired size and fill color if fill is True
        if fill:
            _image_fill: MatLike = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            _image_fill[:] = fill

            # calculate the position to paste the image
            x: int = int((size[0] - _width) / 2)
            y: int = int((size[1] - _height) / 2)

            # calculate the padding
            pad_t: int = y
            pad_b: int = size[1] - _height - y
            pad_l: int = x
            pad_r: int = size[0] - _width - x

            # paste the image on the new image
            _image_fill[y : y + _height, x : x + _width] = _image

            return _image_fill, (_width, _height), (pad_t, pad_b, pad_l, pad_r)

        return _image, (_width, _height), (0, 0, 0, 0)

    # resize the image without keeping the aspect ratio
    return cv2.resize(_image, size, interpolation=cv2.INTER_LANCZOS4), size, (0, 0, 0, 0)


def nms(boxes: NDArray[Any], iou_thresh: float = 0.5) -> tuple[NDArray[np.intp], list[int]] | None:
    # check if there are no boxes
    if len(boxes) == 0:
        return None

    # make a copy of the bounding boxes
    _boxes: NDArray[np.float32] = np.copy(boxes).astype(np.float32)

    # get the coordinates of the bounding boxes [x1, y1, x2, y2]
    x1: NDArray[np.float32] = _boxes[:, 0]
    y1: NDArray[np.float32] = _boxes[:, 1]
    x2: NDArray[np.float32] = _boxes[:, 2]
    y2: NDArray[np.float32] = _boxes[:, 3]

    # compute the area of the bounding boxes
    area: NDArray[Any] = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx: NDArray[np.intp] = np.argsort(y2)

    # initialize the list of picked indices
    indices: list[int] = []

    # keep looping while some indices still remain in the indices list
    while len(idx) > 0:
        # grab the last index in the indices list and add the index value to the list of picked indices
        last: int = len(idx) - 1
        i: int = idx[last]
        indices.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1: NDArray[Any] = np.maximum(x1[i], x1[idx[:last]])
        yy1: NDArray[Any] = np.maximum(y1[i], y1[idx[:last]])
        xx2: NDArray[Any] = np.minimum(x2[i], x2[idx[:last]])
        yy2: NDArray[Any] = np.minimum(y2[i], y2[idx[:last]])

        # compute the width and height of the bounding box
        w: NDArray[Any] = np.maximum(0, xx2 - xx1 + 1)
        h: NDArray[Any] = np.maximum(0, yy2 - yy1 + 1)

        # compute the intersection over union (IoU) of the bounding boxes
        overlap: NDArray[Any] = (w * h) / area[idx[:last]]

        # delete all indices from the index list that have an IoU greater than the provided threshold
        idx = np.delete(idx, np.concatenate(([last], np.where(overlap > iou_thresh)[0])))

    # return the bounding boxes (in integer) that were picked
    return (
        boxes[indices].astype(np.intp),
        indices,
    )


def draw_bbox(
    image: MatLike,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    colors: tuple[int, int, int] = (145, 83, 249),
    thickness: int = 2,
) -> MatLike:
    # check if the labels are provided
    if labels is None:
        labels = ["" for _ in range(len(boxes))]

    # loop over the bounding boxes
    for box, label in zip(boxes, labels):
        # unpack the bounding box
        x1, y1, x2, y2 = box

        # draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), colors, thickness)

        # draw the label with black text and color as background
        if label:
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                colors,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # black color for text
                1,
            )

    return image
