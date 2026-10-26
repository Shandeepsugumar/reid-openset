"""Image transforms and data augmentation pipelines for person ReID."""

from typing import Tuple, Union
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transforms(
    image_size: Union[Tuple[int, int], list] = (256, 128),
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    random_erase_prob: float = 0.5,
) -> transforms.Compose:
    """Build preprocessing and augmentation pipeline.

    Args:
        image_size: Target image size (height, width). Default: (256, 128).
        is_train: If True, applies random horizontal flip, padding+crop, and random erasing.
        mean: Normalization mean for RGB channels.
        std: Normalization std for RGB channels.
        random_erase_prob: Probability of applying random erasing during training.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    height, width = image_size

    if is_train:
        transform_list = [
            transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if random_erase_prob > 0.0:
            transform_list.append(
                transforms.RandomErasing(
                    p=random_erase_prob,
                    value="random",
                )
            )
        return transforms.Compose(transform_list)

    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
