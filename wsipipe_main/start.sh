#!/bin/bash
# https://systems.wiki.cs.st-andrews.ac.uk/index.php/Docker for docker usage
# Pytorch can run GPU accelarated in venv without need for docker - docker provides more flexibility though throws error with cuda (cu113) torch installs

python3 -m venv pytorchvenv
source pytorchvenv/bin/activate
pip install --upgrade pip
pip install --upgrade torch
pip install --upgrade torchvision
pip install --upgrade pandas
pip install --upgrade openslide-python
pip install --upgrade scikit-learn
pip install --upgrade opencv-python
pip install --upgrade Click
pip install --upgrade scipy
pip install --upgrade scikit-image
pip install --upgrade numpy
pip install --upgrade Pillow
pip install --upgrade pylibCZIrw
pip install --upgrade matplotlib

python3 ECdata_processing.py
