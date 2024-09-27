"""
Tools für die App zum Aufnehmen von Trainingsbildern
F.Pingel, 18.03.2024
"""
import threading                # threading für die Arbeit mit Threads



class ImageSaver(threading.Thread):
    """Klasse, die Bilder in einem eigenen Thread speichert."""

    def __init__(self, app):
        """Initialisiert den ImageSaver mit der App-Instanz."""
        super().__init__()
        self.app = app
        self.stop_event = threading.Event()

    def run(self):
        """Die Hauptfunktion des Threads. Führt die Bildspeicherung aus, bis das Stoppsignal gesetzt wird."""
        while not self.stop_event.is_set():         
            self.app.save_image()                           # Bild speichern
            #self.stop_event.wait(self.app.save_interval)    # Warten gemäß des Speicherintervalls
            #self.stop_event.wait(10)    # Warten gemäß des Speicherintervalls
 
    def stop(self):
        """Setzt das Stoppsignal, um den Thread zu beenden."""
        self.stop_event.set()