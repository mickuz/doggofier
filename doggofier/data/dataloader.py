"""This module implements a function to prepare data loaders which enable
training in batches.
"""

import torchvision.transforms as transforms
from typing import List, Dict, Union
from torch.utils.data import DataLoader, random_split
from data.dataset import DogsDataset


VALID_TYPES = {'train', 'val', 'test'}

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def fetch_dataloader(
        data_types: List[str],
        data_dir: str,
        val_size: float,
        params: Dict[str, Union[int, float]]
) -> Dict[str, DataLoader]:
    """Retrieves the data loaders.

    Parameters
    ----------
    data_types : List[str]
        Specifies which dataset splits should be used. It can contain only
        'train', 'val' or 'test' values.
    data_dir : str
        The directory where the data is stored.
    val_size : float
        Proportion of validation and test split size, value between 0 and 1.
    params : Dict[str, Union[int, float]]
        The hyperparameters.

    Returns
    -------
    Dict[str, DataLoader]
        The prepared data loaders.

    Raises
    ------
    ValueError
        When list of data types contains inappropriate values.
    """
    train_dataset = DogsDataset(
        data_dir,
        train=True,
        transform=train_transform
    )
    eval_dataset = DogsDataset(
        data_dir,
        train=False,
        transform=eval_transform
    )

    val_dataset, test_dataset = random_split(
        eval_dataset,
        [
            val_part := int(val_size * len(eval_dataset)),
            len(eval_dataset) - val_part
        ]
    )

    dataloaders = {}
    for data_type in set(data_types):
        if data_type not in VALID_TYPES:
            raise ValueError('The data type must be `train`, `val` or `test`!')
        elif data_type == 'train':
            loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=params['num_workers']
            )
        elif data_type == 'val':
            loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=params['num_workers']
            )
        else:
            loader = DataLoader(
                test_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=params['num_workers']
            )
        dataloaders[data_type] = loader

    return dataloaders
