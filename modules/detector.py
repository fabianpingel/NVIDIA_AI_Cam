import torch
from torch import nn
import cv2
import numpy as np
import logging  
from pathlib import Path
from omegaconf import DictConfig
from collections.abc import Sequence
from typing import Any


from anomalib import TaskType
from anomalib.data import LabelName
from anomalib.data.utils import read_image
from anomalib.data.utils.boxes import masks_to_boxes
from anomalib.utils.visualization import ImageResult

from anomalib.deploy.inferencers.base_inferencer import Inferencer

#from anomalib.models import Patchcore  # Beispielmodell
#from anomalib.data.utils import transform_image
#from torchvision import transforms

#from anomalib.deploy.inferencers import TorchInferencer
#from anomalib.utils.visualization import ImageVisualizer
#from anomalib import TaskType

#from anomalib.deploy.inferencers import TorchInferencer

#torch.set_grad_enabled(mode=False)

class Detector(Inferencer):
    def __init__(self, 
                 path: str | Path,
                 device: str = "auto") -> None:
        """
        Initialisiert den Detector mit einem vortrainierten Anomalie-Modell.
        """
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        
        # Device
        self.device = self._get_device(device)
        self.logger.info( f'Device: {self.device }')

        # Modell laden
        self.checkpoint = self._load_checkpoint(path)
        self.model = self.load_model(path)
        self.metadata = self._load_metadata(path)


        self.predictions = None  # Platzhalter für die Vorhersagen

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Get the device to use for inference.

        Args:
            device (str): Device to use for inference. Options are auto, cpu, cuda.

        Returns:
            torch.device: Device to use for inference.
        """
        if device not in ("auto", "cpu", "cuda", "gpu"):
            msg = f"Unknown device {device}"
            raise ValueError(msg)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)


    def _load_checkpoint(self, path: str | Path) -> dict:
        """Load the checkpoint.

        Args:
            path (str | Path): Path to the torch ckpt file.

        Returns:
            dict: Dictionary containing the model and metadata.
        """
        if isinstance(path, str):
            path = Path(path)

        if path.suffix not in (".pt", ".pth"):
            msg = f"Unknown torch checkpoint file format {path.suffix}. Make sure you save the Torch model."
            raise ValueError(msg)

        return torch.load(path, map_location=self.device)

    def _load_metadata(self, path: str | Path | dict | None = None) -> dict | DictConfig:
        """Load metadata from file.

        Args:
            path (str | Path | dict): Path to the model pt file.

        Returns:
            dict: Dictionary containing the metadata.
        """
        metadata: dict | DictConfig

        if isinstance(path, dict):
            metadata = path
        elif isinstance(path, str | Path):
            checkpoint = self._load_checkpoint(path)

            # Torch model should ideally contain the metadata in the checkpoint.
            # Check if the metadata is present in the checkpoint.
            if "metadata" not in checkpoint:
                msg = (
                    "``metadata`` is not found in the checkpoint. Please ensure that you save the model as Torch model."
                )
                raise KeyError(
                    msg,
                )
            metadata = checkpoint["metadata"]
        else:
            msg = f"Unknown ``path`` type {type(path)}"
            raise TypeError(msg)

        return metadata


    def load_model(self, path: str | Path) -> nn.Module:
        """Load the PyTorch model.

        Args:
            path (str | Path): Path to the Torch model.

        Returns:
            (nn.Module): Torch model.
        """
        checkpoint = self._load_checkpoint(path)
        if "model" not in checkpoint:
            msg = "``model`` is not found in the checkpoint. Please check the checkpoint file."
            raise KeyError(msg)

        model = checkpoint["model"]
        model.eval()
        return model.to(self.device)
    
    
    def predict(
        self,
        image: str | Path | torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> ImageResult:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        if metadata is None:
            metadata = self.metadata if hasattr(self, "metadata") else {}
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=True)

        metadata["image_shape"] = image.shape[-2:]

        processed_image = self.pre_process(image)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, metadata=metadata)

        return ImageResult(
            image=(image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
            pred_score=output["pred_score"],
            pred_label=output["pred_label"],
            anomaly_map=output["anomaly_map"],
            pred_mask=output["pred_mask"],
            pred_boxes=output["pred_boxes"],
            box_labels=output["box_labels"],
        )


    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        
        try:
            # Überprüfen, ob das Bild korrekt geladen wurde
            if image is None:
                raise ValueError(f"Das Bild konnte nicht geladen werden")
            
            # Konvertieren in Graustufen (für die Kreis-Erkennung)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Finde Kreise im Bild mit Hough-Transformation
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1.5,  # Inverser Auflösungsfaktor
                minDist=100,  # Minimale Distanz zwischen den Zentren der Kreise
                param1=300,  # Parameter für den Canny-Edge-Detektor (HoughCircles)
                param2=30,   # Schwellenwert für die Kreiserkennung
                minRadius=250,  # Minimale Kreisgröße
                maxRadius=350   # Maximale Kreisgröße
            )
            
            # Überprüfen, ob Kreise gefunden wurden
            if circles is not None:
                # Konvertiere die Koordinaten und den Radius in Ganzzahlen
                circles = np.round(circles[0, :]).astype("int") 

                # Nehme den ersten gefundenen Kreis (Annahme: es gibt nur einen Kreis)
                (x, y, r) = circles[0]
                
                # Festgelegter Radius zum Zuschneiden
                r = 350

                # Sicherstellen, dass der Ausschnitt im Bildbereich liegt
                if y - r >= 0 and y + r <= image.shape[0] and x - r >= 0 and x + r <= image.shape[1]:
                    # Schneide das rechteckige Stück um das Zentrum des Kreises aus
                    #cropped = image[y - r:y + r, x - r:x + r]
                    cropped = gray[y - r:y + r, x - r:x + r]
                    cropped = cv2.resize(cropped, (416, 416))

                    return cropped
                
                else:
                    self.logger.info(" Ausschnitt liegt außerhalb des Bildbereichs. Rückgabe des Originalbildes.")
            
            else:
                    self.logger.info(" Keine Kreise gefunden. Rückgabe des Originalbildes.")

        # Fehlerbehandlung
        except ValueError as val_error:
            self.logger.error(f' {val_error}')
        except Exception as e:
            self.logger.error(f' Ein unerwarteter Fehler ist aufgetreten: {e}')
        
        # Rückgabe des Originalbildes, wenn kein Kreis gefunden wurde oder Fehler auftreten
        return image
        



    def pre_process(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Vorverarbeitung des Bildes: Konvertiert das Bild in Graustufen, sucht nach Kreisen, 
        und schneidet das Bild basierend auf den gefundenen Kreisen zu.
        
        :param image: Eingabebild als NumPy-Array (im BGR-Format).
        :return: Ausgeschnittenes Bild basierend auf den Kreisen oder das Originalbild, 
                 falls keine Kreise gefunden werden oder ein Fehler auftritt.
        """        

        #Bild mit OpenCV vorverarbeiten
        image = self.prepare_image(image)
        print(image.shape)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if len(image) == 3:
            pass
        
        image = image.unsqueeze(0)

        #if isinstance(image, np.ndarray):
        #    image = torch.from_numpy(image)

        # Rückgabe des Originalbildes, wenn kein Kreis gefunden wurde oder Fehler auftreten
        return image.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward-Pass input tensor to the model.

        Args:
            image (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)


    def post_process(
        self,
        predictions: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor],
        metadata: dict | DictConfig | None = None,
    ) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (Tensor | list[torch.Tensor] | dict[str, torch.Tensor]): Raw output predicted by the model.
            metadata (dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, str | float | np.ndarray]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        # Some models return a Tensor while others return a list or dictionary. Handle both cases.
        # TODO(ashwinvaidya17): Wrap this post-processing stage within the model's forward pass.
        # CVS-122674

        # Case I: Predictions could be a tensor.
        if isinstance(predictions, torch.Tensor):
            anomaly_map = predictions.detach().cpu().numpy()
            pred_score = anomaly_map.reshape(-1).max()

        # Case II: Predictions could be a dictionary of tensors.
        elif isinstance(predictions, dict):
            if "anomaly_map" in predictions:
                anomaly_map = predictions["anomaly_map"].detach().cpu().numpy()
            else:
                msg = "``anomaly_map`` not found in the predictions."
                raise KeyError(msg)

            if "pred_score" in predictions:
                pred_score = predictions["pred_score"].detach().cpu().numpy()
            else:
                pred_score = anomaly_map.reshape(-1).max()

        # Case III: Predictions could be a list of tensors.
        elif isinstance(predictions, Sequence):
            if isinstance(predictions[1], (torch.Tensor)):
                pred_score, anomaly_map = predictions
                anomaly_map = anomaly_map.detach().cpu().numpy()
                pred_score = pred_score.detach().cpu().numpy()
            else:
                pred_score, anomaly_map = predictions
                pred_score = pred_score.detach()
        else:
            msg = (
                f"Unknown prediction type {type(predictions)}. "
                "Expected torch.Tensor, list[torch.Tensor] or dict[str, torch.Tensor]."
            )
            raise TypeError(msg)

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        pred_label: LabelName | None = None
        if "image_threshold" in metadata:
            pred_idx = pred_score >= metadata["image_threshold"]
            pred_label = LabelName.ABNORMAL if pred_idx else LabelName.NORMAL

        pred_mask: np.ndarray | None = None
        if "pixel_threshold" in metadata:
            pred_mask = (anomaly_map >= metadata["pixel_threshold"]).squeeze().astype(np.uint8)

        anomaly_map = anomaly_map.squeeze()
        anomaly_map, pred_score = self._normalize(anomaly_maps=anomaly_map, pred_scores=pred_score, metadata=metadata)

        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
            image_height = metadata["image_shape"][0]
            image_width = metadata["image_shape"][1]
            anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

            if pred_mask is not None:
                pred_mask = cv2.resize(pred_mask, (image_width, image_height))

        if self.metadata["task"] == TaskType.DETECTION:
            pred_boxes = masks_to_boxes(torch.from_numpy(pred_mask))[0][0].numpy()
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }







