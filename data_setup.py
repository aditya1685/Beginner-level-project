import os 
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from PIL import Image
import pathlib


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        try:
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error applying transformations to image {self.paths[index]}: {e}")
        return img, class_idx


def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform: transforms.Compose,
    batch_size:int,
    num_workers: int=0
):
    train_data = ImageFolderCustom(train_dir, transform = transform)
    test_data = ImageFolderCustom(test_dir, transform = transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(train_data,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 num_workers = num_workers,
                                 pin_memory = True)
    test_dataloader = DataLoader(test_data,
                                 batch_size = batch_size,
                                 shuffle = False,
                                 num_workers = num_workers,
                                 pin_memory = True)
    return train_dataloader, test_dataloader, class_names


