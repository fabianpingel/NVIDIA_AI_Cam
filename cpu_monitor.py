"""
Script zum Aufnehmen von CPU Temperatur und Frequenz
F.Pingel, 16.08.2024
"""

import os
import time
import logging
import argparse
import datetime

# Konfigurieren des Loggings auf INFO-Ebene
logging.basicConfig(level=logging.WARNING)
# Einrichten des Loggers für dieses Skript
logger = logging.getLogger(__name__)


def make_parser():
    """
    Erstellt einen Argument-Parser für die Befehlszeilenargumente.
    """

    # Parser erstellen
    parser = argparse.ArgumentParser()

    # Befehlszeilenargumente hinzufügen
    parser.add_argument('--interval', type=int, default=60, help="Intervall zum Speichern der Daten")
    return parser


def get_cpu_temp():
    """Liest die CPU-Temperatur aus."""
    temp = os.popen("vcgencmd measure_temp").readline()
    return float(temp.replace("temp=", "").replace("'C\n", ""))


def get_cpu_freq():
    """Liest die CPU-Taktfrequenz aus."""
    freq = os.popen("vcgencmd measure_clock arm").readline()
    return int(freq.replace("frequency(48)=", "").strip())


def log_data(filename, interval=60):
    """Protokolliert die CPU-Temperatur und -Taktfrequenz in eine Datei."""
    with open(filename, 'a') as file:
        while True:
            # Hole aktuelle Zeit
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Hole CPU-Daten
            cpu_temp = get_cpu_temp()
            cpu_freq = get_cpu_freq() / 1_000_000  # Frequenz in MHz
            
            # Log-Eintrag
            log_entry = f"{timestamp}, Temp: {cpu_temp}°C, Freq: {cpu_freq} MHz\n"
            print(log_entry.strip())
            file.write(log_entry)
            
            # Warte das definierte Intervall
            time.sleep(interval)


def main():
    # Befehlszeilenargumente parsen
    opt = make_parser().parse_args()

    # Informationen über die Befehlszeilenargumente protokollieren
    logger.info(f' Befehlszeilenargumente: {opt}')

    # Zeitstempel für Dateinamen holen
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Monitoring starten
    log_data(f"./logs/cpu_log_{timestamp}.txt", 
             opt.interval)  



if __name__ == "__main__":
    main()