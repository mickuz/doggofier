import os
import json
import xml.etree.ElementTree as etree
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple, Dict


class DogsDataset(Dataset):
    cat_dict = None

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._init_dataset(train)

        if DogsDataset.cat_dict is None:
            DogsDataset.cat_dict = DogsDataset.create_categories(root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
        name = 'train.lst' if train else 'validation.lst'
        with open(os.path.join(self.root, name), 'r', encoding='utf8') as file:
            self.paths = file.readlines()
        self.paths = [path.split('//')[-1].rstrip() for path in self.paths]

    def _extract_annotation(self, path: str) -> Tuple[str, Tuple[int]]:
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
        folders = os.listdir(os.path.join(data_dir, 'annotations'))
        categories = [folder.split('-')[-1].replace('_', ' ').capitalize()
                      for folder in folders]

        cat_dict = {i: cat for i, cat in enumerate(categories)}

        return cat_dict

    @staticmethod
    def category_to_id(cat_dict: Dict[int, str], category: str) -> int:
        category_keys = list(cat_dict.keys())
        category_values = list(cat_dict.values())

        return category_keys[category_values.index(category)]

    @staticmethod
    def save_categories(cat_dict: Dict[int, str], path: str) -> None:
        with open(path, mode='w') as cat_file:
            json.dump(cat_dict, cat_file)
