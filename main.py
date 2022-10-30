import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
from typing import Dict
import pandas as pd
import camelyon17_dataset
from wsipipe.load.datasets.camelyon16 import Camelyon16Loader


if __name__ == "__main__":
    train_dataset = camelyon17_dataset.training(cam17_path='~/data/ec259/camelyon17/raw')
    dset_loader = Camelyon16Loader()
