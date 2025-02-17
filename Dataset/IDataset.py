import os
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import utils.MFileSystem as mf

class IDataset(Dataset):
    pairs = []
    transform = None

    def __init__(self, root_dir, set_name='train_ship', transform=None, image_path='images', label_path='annfiles',image_bit=8, classes=None):
        """
        Description:
            This is an interface for the dataset class. It is used to define the structure of the dataset class.
        Args:
            data: The data that is to be loaded.
            label: The labels of the data.
            transform: The transformations that are to be applied to the data.
        """
        self.transform = transform

        self.root_dir = root_dir
        self.set_name = set_name
        self.image_range = pow(2, image_bit)-1

        images = mf.GetFiles(os.path.join(root_dir, image_path), extension='.png', isRecursive=False)
        annfiles = mf.GetFiles(os.path.join(root_dir, label_path), extension='.geojson', isRecursive=False)

        self.pairs = mf.Pairing(images, annfiles)

        if classes is None:
            self.classes = {"other":0}
        else:
            if not isinstance(classes, dict):
                return ValueError(f"Classes should be a dictionary. Got classes type is '{type(classes)}'")

            for key in classes:
                if not isinstance(classes[key], int):
                    return ValueError(f"Classes should be a dictionary of integers. Got classes type is '{type(classes[key])}'")

            self.classes = classes

        return #NotImplementedError("This is an interface. Please implement the class.")

    def __getitem__(self,index):
        """
        Description:
            This function is used to get the data and label at a particular index.
        Args:
            index: The index at which the data and label is to be fetched.
        Returns:
            data: The data at the given index.
            label: The label at the given index.
        """
        return NotImplementedError("This is an interface. Please implement the class.")

    def __len__(self):
        """
        Description:
            This function is used to get the length of the dataset.
        Returns:
            length: The length of the dataset.
        """
        return len(self.pairs)

    def num_classes(self):
        """
        Description:
            This function is used to get the number of classes.
        Returns:
            num_classes: The number of classes.
        """
        return len(self.classes)

    def image_aspect_ratio(self, index):
        """
        Description:
            This function is used to get the aspect ratio of the image.
        Args:
            index: The index of the image.
        Returns:
            aspect_ratio: The aspect ratio of the image.
        """
        return NotImplementedError("This is an interface. Please implement the class.")

