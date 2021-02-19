import os
import xml.etree.ElementTree as etree
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple, Dict


class DogsDataset(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:

        self.root = root
        self.categories = self._create_categories()
        self.transform = transform
        self.target_transform = target_transform
        self._init_dataset(train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path = os.path.join(self.root, 'images', self.paths[index])
        annot_path = os.path.join(self.root, 'annotations',
                                  self.paths[index] + '.xml')

        image = Image.open(image_path).convert('RGB')

        label, bbox = self._extract_annotation(annot_path)

        image = image.crop(bbox)
        target = self._category_to_id(label)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _init_dataset(self, train: bool) -> None:
        name = 'train.lst' if train else 'validation.lst'
        with open(os.path.join(self.root, name), 'r') as file:
            self.paths = file.readlines()
        self.paths = [path.split('//')[-1].rstrip() for path in self.paths]

    def _create_categories(self) -> Dict[int, str]:
        folders = os.listdir(os.path.join(self.root, 'annotations'))
        categories = [folder.split('-')[-1].replace('_', ' ').capitalize()
                      for folder in folders]

        cat_dict = {i: cat for i, cat in enumerate(categories)}

        return cat_dict

    def _category_to_id(self, category: str) -> int:
        category_keys = list(self.categories.keys())
        category_values = list(self.categories.values())

        return category_keys[category_values.index(category)]

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

    def get_categories(self) -> Dict[int, str]:
        return self.categories
