import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import cv2
import numpy as np

from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.data.utils.image import save_image, show_image
from anomalib.deploy import TorchInferencer
from anomalib.utils.visualization import ImageVisualizer
from torchvision.transforms.v2.functional import to_dtype, to_image

logger = logging.getLogger(__name__)

class Detector():
    def __init__(self, 
                 path: str | Path,
                 device: str = "auto") -> None:
        """
        Initialisiert den Detector mit einem vortrainierten Anomalie-Modell.
        """
        torch.set_grad_enabled(mode=False)

        # Create the inferencer and visualizer.
        self.inferencer = TorchInferencer(path=path, 
                                     device=device)
        
        self.visualizer = ImageVisualizer(mode='full',
                                     task='classification')


        self.predictions = None  # Platzhalter für die Vorhersagen


    def infer(self, image):
        processed_image = self.pre_process(image)

        image_tensor = to_dtype(to_image(processed_image), torch.float32, scale=True)
        predictions = self.inferencer.predict(image=image_tensor)
        output = self.visualizer.visualize_image(predictions)

        return output

    

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        
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

                    # Bild in dreikanalig umwandeln (416, 416) --> (416, 416, 3)
                    gray_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

                    # Resize
                    #cropped = cv2.resize(cropped, (416, 416))
                    gray_image = cv2.resize(gray_image, (416, 416))

                    # return cropped
                    return gray_image
                
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
        



