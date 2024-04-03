import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore
from cv2.typing import MatLike
from numpy.typing import NDArray
from onnxruntime import InferenceSession, SessionOptions  # type: ignore
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from sarkinkofa.constants import MODEL_DIR

from .types import SarkiResult
from .utils import draw_bbox, nms, resize, xywh2xyxy


class SarkiBASE:
    def __init__(self, model_path: Path, names_path: Path) -> None:
        self._model_path: Path = model_path
        self._names_path: Path = names_path

        # check if the model and names files exist
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file {self._model_path} does not exist!")

        if not self._names_path.exists():
            raise FileNotFoundError(f"Names file {self._names_path} does not exist!")

        # initialize the model
        self._initialize()

    @property
    def names(self) -> dict[int, str]:
        return self._names

    def detect(
        self,
        image: MatLike,
        conf_thresh: float = 0.7,
        iou_thresh: float = 0.5,
        inc_cls: list[int] | None = None,
        exc_cls: list[int] | None = None,
        render: bool = False,
    ) -> SarkiResult | None:
        # preprocess the input
        _input: NDArray[Any] = self._preprocess(image)

        # run the inference
        output: Any = self._model.run([self._output_name], {self._input_name: _input})  # type: ignore

        # postprocess the output
        _output: SarkiResult | None = self._postprocess(output, conf_thresh, iou_thresh)

        # check if exc_cls and inc_cls are both provided
        if inc_cls is not None and exc_cls is not None:
            raise ValueError("inc_cls and exc_cls cannot be provided at the same time!")

        # filter the output if required
        if inc_cls is not None and _output is not None:
            _idx: list[int] = [i for i, c in enumerate(_output.cls) if c in inc_cls]

            if len(_idx) > 0:
                _output.boxes = [_output.boxes[i] for i in _idx]
                _output.confs = [_output.confs[i] for i in _idx]
                _output.cls = [_output.cls[i] for i in _idx]
                _output.labels = [_output.labels[i] for i in _idx]
            else:
                return None

        if exc_cls is not None and _output is not None:
            _idx: list[int] = [i for i, c in enumerate(_output.cls) if c not in exc_cls]

            if len(_idx) > 0:
                _output.boxes = [_output.boxes[i] for i in _idx]
                _output.confs = [_output.confs[i] for i in _idx]
                _output.cls = [_output.cls[i] for i in _idx]
                _output.labels = [_output.labels[i] for i in _idx]
            else:
                return None

        # render the output if required
        if render and _output is not None:
            # draw the bounding box
            _image: MatLike = draw_bbox(
                image=image,
                boxes=_output.boxes,
                labels=_output.labels,
            )

            _output.img = _image

        return _output

    def _initialize(self) -> None:
        # load the model
        self._model_opt = SessionOptions()  # type: ignore
        self._model_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # type: ignore
        self._model_opt.intra_op_num_threads = 3  # type: ignore
        self._model_opt.inter_op_num_threads = 3  # type: ignore

        self._model = InferenceSession(
            self._model_path,
            options=self._model_opt,  # type: ignore
            providers=ort.get_available_providers(),  # type: ignore
        )

        # load the names as dict
        self._names: dict[int, str] = {
            idx: name for idx, name in enumerate(self._names_path.read_text().splitlines())
        }
        # get input and output details
        model_inputs: Any = self._model.get_inputs()
        model_outputs: Any = self._model.get_outputs()

        self._input_name: str = model_inputs[0].name
        self._input_shape: tuple[int, int] = (model_inputs[0].shape[3], model_inputs[0].shape[2])
        self._output_name: str = model_outputs[0].name

    def _preprocess(self, image: MatLike) -> NDArray[Any]:
        # set the output shape
        self._output_shape: tuple[int, int] = (image.shape[1], image.shape[0])

        # resize the image to the input shape of the model
        _resize_output: tuple[MatLike, tuple[int, int], tuple[int, int, int, int]] = resize(
            image, self._input_shape, fill=(0, 0, 0)
        )
        _image: MatLike = _resize_output[0]
        self._image_shape: tuple[int, int] = _resize_output[1]
        self._image_pad: tuple[int, int, int, int] = _resize_output[2]

        # normalize the image
        image_arr = np.array(_image, dtype=np.float16) / 255.0

        # transpose the image: HWC to CHW
        image_arr: NDArray[Any] = np.transpose(image_arr, (2, 0, 1))

        # add a batch dimension: CHW to NCHW
        image_tensor: NDArray[Any] = image_arr[np.newaxis, :, :, :].astype(np.float16)

        return image_tensor

    def _postprocess(
        self,
        output: Any,
        conf_thresh: float = 0.7,
        iou_thresh: float = 0.5,
    ) -> SarkiResult | None:
        # remove the batch dimension
        preds: NDArray[Any] = np.squeeze(output[0]).T
        boxes: NDArray[Any] = preds[:, :4]
        cls: NDArray[Any] = np.argmax(preds[:, 4:], axis=1)
        confs: NDArray[Any] = np.max(preds[:, 4:], axis=1)

        # get the class with the highest probability
        preds = preds[confs > conf_thresh, :]
        boxes = boxes[confs > conf_thresh, :]
        cls = cls[confs > conf_thresh]
        confs = confs[confs > conf_thresh]

        if len(preds) == 0:
            return None

        # remove the padding
        boxes[:, 0] -= self._image_pad[2]
        boxes[:, 1] -= self._image_pad[0]

        # rescale the bounding boxes to the original image size
        boxes = np.divide(
            boxes,
            [
                self._image_shape[0],
                self._image_shape[1],
                self._image_shape[0],
                self._image_shape[1],
            ],
        )
        boxes = np.multiply(
            boxes,
            [
                self._output_shape[0],
                self._output_shape[1],
                self._output_shape[0],
                self._output_shape[1],
            ],
        )

        # convert from [x, y, w, h] to [x1, y1, x2, y2]
        boxes = xywh2xyxy(boxes)

        # compute the non-maxima suppression
        _cls: NDArray[np.intp] = np.empty((0), dtype=np.intp)
        _boxes: NDArray[np.intp] = np.empty((0, 4), dtype=np.intp)
        _confs: NDArray[np.float32] = np.empty((0), dtype=np.float32)
        _labels: list[str] = []
        _idx: list[int] = []

        for _class in np.unique(cls):
            class_indices: NDArray[Any] = np.where(cls == _class)[0]

            _nms_output: tuple[NDArray[np.intp], list[int]] | None = nms(
                boxes[class_indices, :], iou_thresh
            )

            if _nms_output is not None:
                _boxes = np.concatenate((_boxes, _nms_output[0]))
                _confs = np.concatenate((_confs, confs[class_indices][_nms_output[1]]))
                _cls = np.concatenate((_cls, cls[class_indices][_nms_output[1]]))
                _labels.extend([self._names[int(i)] for i in cls[class_indices][_nms_output[1]]])
                _idx.extend(class_indices[_nms_output[1]])

        # return the bounding boxes (as list[list[int]]), the class indices (as list[int]),
        # and the class confidences (as list[float])
        if len(_idx) == 0:
            return None

        # sort the bounding boxes by their x1 coordinate
        _idx = np.argsort(_boxes[:, 0]).tolist()
        _boxes = _boxes[_idx]
        _confs = _confs[_idx]
        _cls = _cls[_idx]
        _labels = [_labels[i] for i in _idx]

        return SarkiResult(
            img=None,
            boxes=_boxes.tolist(),
            confs=_confs.tolist(),
            cls=_cls.tolist(),
            labels=_labels,
        )


class SarkiANPD(SarkiBASE):
    def __init__(self) -> None:
        super().__init__(
            model_path=Path(MODEL_DIR) / "anpd.onnx", names_path=Path(MODEL_DIR) / "anpd.names"
        )


class SarkiANPR(SarkiBASE):
    def __init__(self) -> None:
        super().__init__(
            model_path=Path(MODEL_DIR) / "anpr.onnx", names_path=Path(MODEL_DIR) / "anpr.names"
        )


class SarkiFSWatcher:
    """
    A class for watching a folder for changes and triggering an event handler.

    Attributes:
        input_folder (str): The path to the input folder.
        output_folder (str): The path to the output folder.
        observer (Observer): The observer object used to watch the input folder.
        detector (SARKINkofa): The detector object used to detect changes in the input folder.
        event_handler (SARKINKofaFSHandler): The event handler object used to handle changes in the input folder.
        verbose (bool): Whether to print verbose output.
    """

    def __init__(
        self,
        input_folder: str | list[str],
        ev_handler: FileSystemEventHandler,
        frequency: float = 0.5,
        recursive: bool = False,
        **kwargs: Any,
    ) -> None:
        self.input_folder: list[str] = (
            input_folder if isinstance(input_folder, list) else [input_folder]
        )
        self.observer: BaseObserver = Observer()
        self.handler: FileSystemEventHandler = ev_handler
        self.frequency: float = frequency
        self.recursive: bool = recursive

    def start_watching(self) -> None:
        """
        Starts watching the input folder for any changes and triggers the event handler
        """
        for folder in self.input_folder:
            self.observer.schedule(self.handler, folder, recursive=self.recursive)  # type: ignore

        self.observer.start()

        try:
            while True:
                time.sleep(self.frequency)
        except KeyboardInterrupt:
            self.stop_watching()

    def stop_watching(self) -> None:
        """
        Stops the observer and waits for it to finish.
        """
        self.observer.stop()
        self.observer.join()
