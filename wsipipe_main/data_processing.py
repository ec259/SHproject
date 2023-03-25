import time
import wsipipe.datasets.camelyon17 as camelyon17
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
from wsipipe.load.datasets.registry import register_loader
from wsipipe.load.datasets.camelyon17 import Camelyon17Loader
from wsipipe.preprocess.patching.patchset_utils import load_patchsets_from_directory, combine
from wsipipe.preprocess.sample.sampler import balanced_sample
from pathlib import Path

root = Path('/data/ec259/camelyon17/raw/')
data = camelyon17.training(cam17_path=root)
register_loader(Camelyon17Loader)
dset_loader = Camelyon17Loader()

# -- SPLITTING INTO TRAIN, TEST, VALIDATION

# split train and test 80/20
train_dset, test_dset = train_test_split(data, test_size=0.2)

# split training into train and validate (80/20)
train_dset, validate_dset = train_test_split(train_dset, test_size=0.2)

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

# -- CREATING PATCHSET --

patchfinder = GridPatchFinder(patch_level=1, patch_size=256, stride=256, labels_level=5)

print("\n----- MAKING PATCHSETS ----- ")
start = time.time()
patchset_dir = root / 'patchsets'

# Patchsets for the whole dataset in individual folders:
training_psets_for_dset = make_and_save_patchsets_for_dataset(
    dataset=train_dset,
    loader=dset_loader,
    tissue_detector=tisdet,
    patch_finder=patchfinder,
    output_dir=patchset_dir / 'training'
)

validation_psets_for_dset = make_and_save_patchsets_for_dataset(
    dataset=validate_dset,
    loader=dset_loader,
    tissue_detector=tisdet,
    patch_finder=patchfinder,
    output_dir=patchset_dir / 'validation'
)

test_psets_for_dset = make_and_save_patchsets_for_dataset(
    dataset=test_dset,
    loader=dset_loader,
    tissue_detector=tisdet,
    patch_finder=patchfinder,
    output_dir=patchset_dir / 'test'
)
print("----- FINISHED MAKING PATCHSETS: " + str(time.time() - start) + "-----")

# -- LOAD ALL PATCHES --
print("\n----- COMBINING AND BALANCING PATCH SETS -----")
start = time.time()
loaded_training_psets = load_patchsets_from_directory(patchset_dir / 'training')
loaded_validation_psets = load_patchsets_from_directory(patchset_dir / 'validation')
loaded_test_psets = load_patchsets_from_directory(patchset_dir / 'test')

# -- COMBINE ALL PATCHES INTO ONE PATCHSET --
combined_training_pset = combine(loaded_training_psets)
combined_validation_pset = combine(loaded_validation_psets)
combined_test_pset = combine(loaded_test_psets)

# -- EXTRACT BALANCED SAMPLE FOR TRAIN AND VALIDATE --
train_balanced_pset = balanced_sample(combined_training_pset, 200000)
valid_balanced_pset = balanced_sample(combined_validation_pset, 200000)
test_balanced_pset = balanced_sample(combined_test_pset, 200000)
print("----- FINISHED BALANCING PATCHSETS: " + str(time.time() - start) + " -----")

# -- EXPORT ALL PATCHES FROM PATCHSET --
print("\n----- CREATING PATCHES FROM PATCHSET -----")
start = time.time()
train_balanced_pset.export_patches(root / 'patches' / "train_patches")
valid_balanced_pset.export_patches(root / 'patches' / "validate_patches")
test_balanced_pset.export_patches(root / 'patches' / "test_patches")
print("----- FINISHED CREATING PATCHES FROM PATCHSET " + str(time.time() - start) + "-----")

