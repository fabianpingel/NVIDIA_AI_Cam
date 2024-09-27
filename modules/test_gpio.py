from gpio import GPIOController
from settings import *
import logging
import time          

# GPIO
gpio_controller = GPIOController()

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Boolean
debug_light = False
debug_motor = False
debug_motor_and_light = True
duration = 60 # Zeit in Sekunden für Tests


# Tests für Leuchte
if debug_light:
    # Konfiguriere Pin Leuchte als Output --> für Leuchte
    gpio_controller.setup_pin(GPIO_PIN_LEUCHTE, 'output')
    # Setze Pin Leuchte auf HIGH
    gpio_controller.write_pin(GPIO_PIN_LEUCHTE, True)

    # Leuchtdauer
    time.sleep(duration)

    # Setze Pin Leuchte auf LOW
    gpio_controller.write_pin(GPIO_PIN_LEUCHTE, False)


# Tests für Motor
if debug_motor:
    # Konfiguriere Pin Motor als Output --> für Leuchte
    gpio_controller.setup_pin(GPIO_PIN_MOTOR, 'output')
    # Setze Pin Motor auf HIGH
    gpio_controller.write_pin(GPIO_PIN_MOTOR, True)

    # Zeit für die Umdrehung
    time.sleep(duration)

    # Setze Pin Motor auf LOW
    gpio_controller.write_pin(GPIO_PIN_MOTOR, False)


# Tests für Motor UND Leuchte
if debug_motor_and_light:
    # Konfiguriere Pin Motor als Output --> für Motor
    gpio_controller.setup_pin(GPIO_PIN_MOTOR, 'output')
    # Setze Pin Motor auf HIGH
    gpio_controller.write_pin(GPIO_PIN_MOTOR, True)

    # Konfiguriere Pin Leuchte als Output --> für Leuchte
    gpio_controller.setup_pin(GPIO_PIN_LEUCHTE, 'output')
    # Setze Pin Leuchte auf HIGH
    gpio_controller.write_pin(GPIO_PIN_LEUCHTE, True)

    # Zeitdauer
    time.sleep(duration)

    # Setze Pin Motor auf LOW
    gpio_controller.write_pin(GPIO_PIN_MOTOR, False)

    # Setze Pin Leuchte auf LOW
    gpio_controller.write_pin(GPIO_PIN_LEUCHTE, False)


# Bereinige alle Pins bei Programmende
gpio_controller.cleanup()
    