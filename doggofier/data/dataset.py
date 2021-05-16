"""This module implements the custom Tsinghua Dogs dataset to make data loading
and preprocessing easier.
"""

import os
import json
import xml.etree.ElementTree as etree
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple, Dict


class DogsDataset(Dataset):
    """The implementation of Tsinghua Dogs dataset. The data should be stored
    in a separate directory. There should be 2 subdirectories: 'images' and
    'annotations'. Those subdirectories should contain a folder for every
    category with images in JPG or JPEG format and annotations in XML format
    respectively. Additionally, the directory with the data needs to have 2 LST
    files with list of training and validation examples.
    """

    cat_dict = None

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        """
        Parameters
        ----------
        root : str
            The directory where the data is stored.
        train : bool, optional
            Specifies if an instance of the dataset is a training split,
            by default True.
        transform : Optional[Callable], optional
            Transforms applied to images, by default None.
        target_transform : Optional[Callable], optional
            Transforms applied to target variable, by default None.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._init_dataset(train)

        if DogsDataset.cat_dict is None:
            DogsDataset.cat_dict = DogsDataset.create_categories(root)

    def __len__(self) -> int:
        """An overriden magic method returning dataset length.

        Returns
        -------
        int
            Number of examples in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """An overriden magic method returning an example with given index.

        Parameters
        ----------
        index : int
            Index of example to return.

        Returns
        -------
        Tuple[Any, Any]
            An example that consists of an image and its label.
        """
        image_path = os.path.join(self.root, 'images', self.paths[index])
        annot_path = os.path.join(self.root, 'annotations',
                                  self.paths[index] + '.xml')

        image = Image.open(image_path).convert('RGB')

        label, bbox = self._extract_annotation(annot_path)

        image = image.crop(bbox)
        target = DogsDataset.category_to_id(DogsDataset.cat_dict, label)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _init_dataset(self, train: bool) -> None:
        """Gives a list of relative paths to the dataset examples.

        Parameters
        ----------
        train : bool
            Specifies if an instance of the dataset is a training split.
        """
        name = 'train.lst' if train else 'validation.lst'
        with open(os.path.join(self.root, name), 'r', encoding='utf8') as file:
            self.paths = file.readlines()
        self.paths = [path.split('//')[-1].rstrip() for path in self.paths]

    def _extract_annotation(self, path: str) -> Tuple[str, Tuple[int]]:
        """Retrieves an information about dog breed and bounding box of the dog
        where bounding box has the following format (xmin, ymin, xmax, ymax) to
        label and crop an image.

        Parameters
        ----------
        path : str
            A relative path to the example.

        Returns
        -------
        Tuple[str, Tuple[int]]
            A label of a dog depicted on the photo and a bounding box to crop
            the area with the dog.
        """
        tree = etree.parse(path)
        root = tree.getroot()
        objects = root.findall('object')

        for obj in objects:
            label = obj.find('name').text.replace('_', ' ').capitalize()

            bodybndbox = obj.find('bodybndbox')
            x1 = int(bodybndbox.find('xmin').text)
            y1 = int(bodybndbox.find('ymin').text)
            x2 = int(bodybndbox.find('xmax').text)
            y2 = int(bodybndbox.find('ymax').text)
            bbox = (x1, y1, x2, y2)

        return label, bbox

    @staticmethod
    def create_categories(data_dir: str) -> Dict[int, str]:
        """Creates a mapping dictionary with dataset labels to simplify
        conversion between category and numeric value.

        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.

        Returns
        -------
        Dict[int, str]
            A mapping of dataset labels to integer values.
        """
        folders = os.listdir(os.path.join(data_dir, 'annotations'))
        categories = [folder.split('-')[-1].replace('_', ' ').capitalize()
                      for folder in folders]

        cat_dict = {i: cat for i, cat in enumerate(categories)}

        return cat_dict

    @staticmethod
    def category_to_id(cat_dict: Dict[int, str], category: str) -> int:
        """Converts a label into numeric value.

        Parameters
        ----------
        cat_dict : Dict[int, str]
            A mapping of dataset labels to integer values.
        category : str
            Label's name.

        Returns
        -------
        int
            A numeric value representing given category.
        """
        category_keys = list(cat_dict.keys())
        category_values = list(cat_dict.values())

        return category_keys[category_values.index(category)]

    @staticmethod
    def save_categories(cat_dict: Dict[int, str], path: str) -> None:
        """Saves a mapping dictionary into JSON file to provide easier access
        from other parts of an application without initializing the dataset.

        Parameters
        ----------
        cat_dict : Dict[int, str]
            A mapping of dataset labels to integer values.
        path : str
            Path to the JSON file where the mapping is going to be saved.
        """
        with open(path, mode='w') as cat_file:
            json.dump(cat_dict, cat_file)
