import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
from typing import Dict
import pandas as pd
import camelyon17_dataset
import wsipipe
from wsipipe.load.datasets.camelyon16 import Camelyon16Loader
from wsipipe.datasets.dataset_utils import sample_dataset
from wsipipe.utils import np_to_pil

if __name__ == "__main__":
    train_dset = camelyon17_dataset.training(cam17_path='~/data/ec259/camelyon17/raw')
    dset_loader = Camelyon16Loader()

    small_train_dset = sample_dataset(train_dset, 2)
    row = small_train_dset.iloc[0]

    # View slide
    with dset_loader.load_slide(row.slide) as slide:
        thumb = slide.get_thumbnail(5)

    from wsipipe.load.annotations import visualise_annotations
    np_to_pil(thumb)

    labelled_image = visualise_annotations(
        row.annotation,
        row.slide,
        dset_loader,
        5,
        row.label
    )
    np_to_pil(labelled_image * 100)
