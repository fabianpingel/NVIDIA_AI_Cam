# RPI_AI_Cam 
🚧 UNDER CONSTRUCTION!!!

Grundlage ist das verlinkte [PyTorch-Tutorial](https://pytorch.org/tutorials/intermediate/realtime_rpi.html)

## Installation
### 1. [RPI Imager](https://www.raspberrypi.com/software/) downloaden: [RPI OS 64bit](https://www.raspberrypi.com/software/operating-systems/) *bullseye* wählen.
### 2. Kamera aktivieren
Sobald das System hochgefahren und die Ersteinrichtung abgeschlossen ist, muss die Datei `/boot/config.txt` bearbeitet werden, um die Kamera zu aktivieren.
```
sudo nano /boot/config.txt
```


```
# This enables the extended features such as the camera.
start_x=1

# This needs to be at least 128M for the camera processing, if it's bigger you can just leave it as is.
gpu_mem=128

# You need to commment/remove the existing camera_auto_detect line since this causes issues with OpenCV/V4L2 capture.
#camera_auto_detect=1
```

### 3. PyTorch und OpenCV installieren

```
pip install torch==1.10.2 torchvision==0.11.2 torchaudio==0.10.1
pip install opencv-python==4.5.5.64
pip install numpy --upgrade
```

Versionen testen: 
```
python -c "import torch; print(torch.__version__)"
```

### 4. Optional
[Visual Studio Code](https://code.visualstudio.com/docs/setup/raspberry-pi) installieren
```
sudo apt update
sudo apt install code
```

[SWAP File](https://pimylifeup.com/raspberry-pi-swap-file/) vergrößern
```
# stop the operating system from using the current swap file
sudo dphys-swapfile swapoff
# open swapfile using nano 
sudo nano /etc/dphys-swapfile
```
```
# modify the numerical value (in megabytes)
CONF_SWAPSIZE=4096
```
```
# re-initialize the RPi’s swap file 
sudo dphys-swapfile setup
# turn the swap back on
sudo dphys-swapfile swapon
# restart
sudo reboot
```

### 5. Sentinel installieren
Das Skript `install_sentinel.sh` führt folgende Schritte aus:

- wechsel ins temporäre Verzeichnis /tmp
- SentinelOne-Agent-Installationsdatei herunterladen
- Sentinel Agenten mit dpkg installieren
-  Management-Token setzen
- SentinelOne-Dienst starten und dessen Status anzeigen

Nachdem das Skript abgeschlossen ist, sollte der SentinelOne Agent korrekt installiert und gestartet sein.
```
# Sicherstellen, dass das Skript ausführbar ist:
chmod +x install_sentinel.sh
```
```
# Skript ausführen
./install_sentinel.sh
```



