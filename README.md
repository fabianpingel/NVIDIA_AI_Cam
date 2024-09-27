# NVIDIA_AI_Cam 
🚧 UNDER CONSTRUCTION!!!


Probleme mit av und OpenCV cv2.imshow() siehe: https://github.com/pytorch/vision/issues/5940

``pip uninstall av``


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



