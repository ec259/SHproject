import datetime

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
from wsipipe.load.annotations import visualise_annotations
from wsipipe.preprocess.tissue_detection import TissueDetectorGreyScale
from wsipipe.preprocess.tissue_detection import SimpleClosingTransform, SimpleOpeningTransform, GaussianBlur
from wsipipe.preprocess.tissue_detection import visualise_tissue_detection_for_slide
from wsipipe.preprocess.patching import GridPatchFinder, make_patchset_for_slide
from wsipipe.preprocess.patching import GridPatchFinder, make_patchset_for_slide
from wsipipe.preprocess.patching import make_patchsets_for_dataset
from wsipipe.preprocess.patching import make_and_save_patchsets_for_dataset
from wsipipe.preprocess.patching import GridPatchFinder, make_patchset_for_slide
from wsipipe.preprocess.patching import visualise_patches_on_slide
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data = camelyon17_dataset.training(cam17_path='/data/ec259/camelyon17/raw')
    train_dset, test_dset = train_test_split(data, test_size=0.2)
    dset_loader = Camelyon16Loader()

    row = train_dset.iloc[0]

    # View slide
    with dset_loader.load_slide(row.slide) as slide:
        thumb = slide.get_thumbnail(5)

    np_to_pil(thumb)

    labelled_image = visualise_annotations(
        row.annotation,
        row.slide,
        dset_loader,
        5,
        row.label
    )
    np_to_pil(labelled_image * 100)

    # -- BACKGROUND SUBTRACTION --

    tisdet = TissueDetectorGreyScale(grey_level=0.85)
    tissmask = tisdet(thumb)
    np_to_pil(tissmask)

    prefilt = GaussianBlur(sigma=2)
    morph = [SimpleOpeningTransform(), SimpleClosingTransform()]
    tisdet = TissueDetectorGreyScale(
        grey_level=0.75,
        morph_transform=morph,
        pre_filter=prefilt
    )
    tissmask = tisdet(thumb)
    np_to_pil(tissmask)

    visualise_tissue_detection_for_slide(row.slide, dset_loader, 5, tisdet)

    # -- PATCH EXTRACTION --

    patchfinder = GridPatchFinder(patch_level=1, patch_size=512, stride=512, labels_level=5)
    pset = make_patchset_for_slide(row.slide, row.annotation, dset_loader, tisdet, patchfinder)

    print("STARTING PATCH EXTRACTION: " + str(datetime.time))

    # Patches for the whole dataset:
    psets_for_dset = make_and_save_patchsets_for_dataset(
        dataset=train_dset,
        loader=dset_loader,
        tissue_detector=tisdet,
        patch_finder=patchfinder,
        output_dir="/data/ec259/camelyon17/raw/patches"
    )

    print("FINISHED PATCH EXTRACTION: " + str(datetime.time))

    # Patches for a single slide
    # patchfinder = GridPatchFinder(patch_level=1, patch_size=512, stride=512, labels_level=5)
    # pset = make_patchset_for_slide(row.slide, row.annotation, dset_loader, tisdet, patchfinder)
    # visualise_patches_on_slide(pset, vis_level=5)
