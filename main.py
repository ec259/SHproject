import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
from typing import Dict
import pandas as pd
import camelyon17_dataset
import wsipipe
from wsipipe.load.annotations import AnnotationSet, load_annotations_asapxml
from wsipipe.load.datasets.loader import Loader
from wsipipe.load.slides import OSSlide, SlideBase
from wsipipe.datasets import camelyon16


if __name__ == "__main__":
    camelyon17_dataset.training(cam17_path="../../data/ec259/camelyon17/raw")
