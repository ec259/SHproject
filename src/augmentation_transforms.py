import torch
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize
from torchvision.transforms import ToTensor, transforms
from torchvision.models import ResNet50_Weights

# Transforms or different augmentations
weights = ResNet50_Weights.DEFAULT
horizontal_flip = transforms.RandomHorizontalFlip(1)
vertical_flip = transforms.RandomVerticalFlip(1)
rotate_90 = transforms.RandomRotation(degrees=(90, 90))
rotate_180 = transforms.RandomRotation(degrees=(180, 180))
rotate_270 = transforms.RandomRotation(degrees=(270, 270))
colour = transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
    )
crop = transforms.RandomResizedCrop(size=(224, 224))

# Different Combinations
combo_1 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip
)

combo_2 = torch.nn.Sequential(
    weights.transforms(),
    rotate_90,
    rotate_180,
    rotate_270
)

combo_3 = torch.nn.Sequential(
    weights.transforms(),
    colour
)

combo_4 = torch.nn.Sequential(
    weights.transforms(),
    crop
)

combo_5 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    rotate_90,
    rotate_180,
    rotate_270
)

combo_6 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    colour
)

combo_7 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    crop
)

combo_8 = torch.nn.Sequential(
    weights.transforms(),
    rotate_90,
    rotate_180,
    rotate_270,
    colour
)

combo_9 = torch.nn.Sequential(
    weights.transforms(),
    rotate_90,
    rotate_180,
    rotate_270,
    crop
)

combo_10 = torch.nn.Sequential(
    weights.transforms(),
    colour,
    crop
)

combo_11 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    rotate_90,
    rotate_180,
    rotate_270,
    colour
)

combo_12 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    rotate_90,
    rotate_180,
    rotate_270,
    crop
)

combo_13 = torch.nn.Sequential(
    weights.transforms(),
    rotate_90,
    rotate_180,
    rotate_270,
    colour,
    crop
)

combo_14 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    colour,
    crop
)

combo_15 = torch.nn.Sequential(
    weights.transforms(),
    horizontal_flip,
    vertical_flip,
    rotate_90,
    rotate_180,
    rotate_270,
    colour,
    crop
)
