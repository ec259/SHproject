#!/bin/bash
python3 -m venv pytorchvenv
source pytorchvenv/bin/activate
pip install --upgrade pip
pip install --upgrade torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install --upgrade wsipipe
pip install --upgrade pandas
pip install --upgrade openslide-python
pip install --upgrade scikit-learn
python3 data_processing.py