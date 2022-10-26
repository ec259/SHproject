"""
This module creates the dataframe for the camelyon 17 dataset with the following columns:
    - The slide column stores the paths on disk of the whole slide images.
    - The annotation column records a path to the annotation files.
    - The label column is the slide level label.
    - The tags column is blank for camelyon 16.
This assumes there is a folder on disk structured the same as downloading
from the camelyon grand challenge Camelyon 17 google drive:
https://camelyon17.grand-challenge.org/Data/
"""

from pathlib import Path

import pandas as pd


def training(cam17_path: Path = Path("data", "camelyon17"), project_root: Path = None) -> pd.DataFrame:
    """ Create Camelyon 17 training dataset
    This function goes through the input directories for the training slides,
    and matches up the annotations and slides.
    It creates a dataframe with slide path with matching annotation path, and slide label.
    There is an empty tags column that is not used for this dataset
    Args:
        cam17_path (Path, optional): a path relative to the project root that is the location
            of the Camelyon 17 data. Defaults to data/camelyon17.
    Returns:
        df (pd.DataFrame): A dataframe with columns: slide, annotation, label and tags
    """
    # Set up the paths to the slides and annotations
    # For Camelyon17, data is split into centers also, need to combine data from all different center's and patients

    if project_root is None:
        dataset_root = Path(cam17_path) / "training"
    else:
        dataset_root = project_root / Path(cam17_path) / "training"
    annotations_dir = dataset_root / "lesion_annotations"
    center_0_slide_dir = dataset_root / "center_0"
    center_1_slide_dir = dataset_root / "center_1"
    center_2_slide_dir = dataset_root / "center_2"
    center_3_slide_dir = dataset_root / "center_3"
    center_4_slide_dir = dataset_root / "center_4"

    # all paths are relative to the project root if defined
    if project_root is None:
        annotation_paths = sorted(
            [p for p in annotations_dir.glob("*.xml")]
        )
        center_0_slide_paths = sorted(
            [p for p in center_0_slide_dir.glob("*.tif")]
        )
        center_1_slide_paths = sorted(
            [p for p in center_1_slide_dir.glob("*.tif")]
        )
        center_2_slide_paths = sorted(
            [p for p in center_2_slide_dir.glob("*.tif")]
        )
        center_3_slide_paths = sorted(
            [p for p in center_3_slide_dir.glob("*.tif")]
        )
        center_4_slide_paths = sorted(
            [p for p in center_4_slide_dir.glob("*.tif")]
        )

    else:
        annotation_paths = sorted(
            [p.relative_to(project_root) for p in annotations_dir.glob("*.xml")]
        )
        center_0_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_0_slide_dir.glob("*.tif")]
        )
        center_1_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_1_slide_dir.glob("*.tif")]
        )
        center_2_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_2_slide_dir.glob("*.tif")]
        )
        center_3_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_3_slide_dir.glob("*.tif")]
        )
        center_4_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_4_slide_dir.glob("*.tif")]
        )

    # turn them into a data frame and pad with empty annotation paths
    # Combine all centers for one directory of all slides
    df = pd.DataFrame()
    # adds all slides regardless of annotation status, gh
    df["slide"] = center_0_slide_paths + center_1_slide_paths + center_2_slide_paths + center_3_slide_paths + center_4_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]

    # Camelyon 17 - we don't know which slides are tumor or normal from a first glance, isn't separated
    # df["label"] = ["tumor"] * len(tumor_slide_paths) + ["normal"] * len(
    #     normal_slide_paths
    # )
    df["tags"] = ""

    return df
