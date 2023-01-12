#!/bin/bash
# https://systems.wiki.cs.st-andrews.ac.uk/index.php/Docker for docker usage
# Pytorch can run GPU accelarated in venv without need for docker - docker provides more flexibility though throws error with cuda (cu113) torch installs

python3 -m venv pytorchvenv
source pytorchvenv/bin/activate
pip install --upgrade pip
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pandas
pip install openslide-python
pip install scikit-learn
pip install opencv-python
pip install Click
pip install scipy
pip install scikit-image
pip install numpy
pip install Pillow
pip install pylibCZIrw
pip install matplotlib

#python3 ECdata_processing.py
python3 ECcnn.py
