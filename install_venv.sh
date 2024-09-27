#!/bin/bash

# Hier den Namen der virtuellen Umgebung angeben
VENV="AI_Cam_venv"

echo "1. Ordner erzeugen"
mkdir $VENV

echo "2. Virtuelle Umgebung erzeugen"
python -m venv $VENV

echo "3. Virtuelle Umgebung aktivieren"
source ./$VENV/bin/activate

echo "4. PIP Upgrade"
python -m pip install --upgrade pip



