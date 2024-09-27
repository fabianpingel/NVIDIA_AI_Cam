import RPi.GPIO as GPIO
import logging
import time


class GPIOController:
    def __init__(self):
        # Verwenden des BCM Pin Nummernschemas
        GPIO.setmode(GPIO.BCM)
        # Liste der initialisierten Pins, um sie später sauber freizugeben
        self.initialized_pins = []
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.WARNING)
        self.logger.info(" GPIO Controller initialisiert")

    def setup_pin(self, pin, mode):
        """Konfiguriert einen GPIO Pin.
        
        Args:
            pin (int): Der GPIO Pin Nummer (BCM Schema).
            mode (str): Der Modus des Pins: 'input' oder 'output'.
        """
        try:
            if mode.lower() == 'input':
                GPIO.setup(pin, GPIO.IN)
            elif mode.lower() == 'output':
                GPIO.setup(pin, GPIO.OUT)
            else:
                raise ValueError("Modus muss 'input' oder 'output' sein.")
            self.initialized_pins.append(pin)
            self.logger.info(f" Pin {pin} als {mode.upper()} konfiguriert")
        except ValueError as e:
            self.logger.error(e)
            raise
        except Exception as e:
            self.logger.error(f" Fehler beim Einrichten des Pins {pin}: {e}")
            raise

    def write_pin(self, pin, state):
        """Setzt den Zustand eines GPIO Pins (nur Output).
        
        Args:
            pin (int): Der GPIO Pin Nummer (BCM Schema).
            state (bool): True für HIGH, False für LOW.
        """
        try:
            if pin not in self.initialized_pins:
                raise RuntimeError("Pin nicht als Output initialisiert.")
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            self.logger.info(f" Pin {pin} auf {'HIGH' if state else 'LOW'} gesetzt")
        except RuntimeError as e:
            self.logger.error(e)
            raise
        except Exception as e:
            self.logger.error(f" Fehler beim Setzen des Pins {pin}: {e}")
            raise

    def read_pin(self, pin):
        """Liest den Zustand eines GPIO Pins (nur Input).
        
        Args:
            pin (int): Der GPIO Pin Nummer (BCM Schema).
            
        Returns:
            bool: True, wenn der Pin HIGH ist, sonst False.
        """
        try:
            if pin not in self.initialized_pins:
                raise RuntimeError("Pin nicht als Input initialisiert.")
            state = GPIO.input(pin)
            self.logger.info(f" Pin {pin} gelesen, Zustand ist {'HIGH' if state else 'LOW'}")
            return state
        except RuntimeError as e:
            self.logger.error(e)
            raise
        except Exception as e:
            self.logger.error(f" Fehler beim Lesen des Pins {pin}: {e}")
            raise

    def cleanup(self):
        """Bereinigt alle initialisierten GPIO Pins."""
        GPIO.cleanup(self.initialized_pins)
        self.initialized_pins = []
        self.logger.info("GPIO Pins bereinigt")