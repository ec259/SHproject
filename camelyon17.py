import wsipipe
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from pathlib import Path
from typing import Dict

from wsipipe.load.annotations import AnnotationSet, load_annotations_asapxml
from wsipipe.load.datasets.loader import Loader
from wsipipe.load.slides import OSSlide, SlideBase