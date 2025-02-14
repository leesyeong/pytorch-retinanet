# import os
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from retinanet.IDataset import IDataset

PAIR_IMG = 0
PAIR_ANN = 1

class MSSDD(IDataset):
    """Sar ship detection dataset dataset."""

    def __init__(self, root_dir, set_name='train_ship', transform=None, image_path='images', label_path='annfiles',image_bit=8, classes=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(MSSDD, self).__init__(root_dir, set_name, transform, image_path, label_path, image_bit, classes=classes)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img = cv2.imread(pair[PAIR_IMG])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float() / self.image_range
            img = img.permute(2, 0, 1)

        # 어노테이션 가져오기
        annots = self.GetShipList(pair[PAIR_ANN], mode='xyxy')
        annots = np.array(annots)  # 1차원 배열을 2차원 배열로 변환
        annots = torch.from_numpy(annots).float()  # 어노테이션을 텐서로 변환

        return {'img': img, 'annot': annots}


    def visualize(self, idx):
        pair = self.pairs[idx]
        img = cv2.imread(pair[PAIR_IMG])

        annots = self.GetShipList(pair[PAIR_ANN])

        for annot in annots:
            x, y, w, h, theta, cls_num = annot
            rect = ((x, y), (w, h), theta)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(img, [box_points], 0, (255, 0, 0), 2)

        cv2.imshow('image', img)
        cv2.waitKey(0)

    @staticmethod
    def GetShipList(file_path, mode:str='xywh'):
        """
        Description:
            This function is used to get the list of ships.
        Returns:
            ship_list: The list of ships.
        """

        json_dict = None

        with open(file_path, 'r') as f:
            json_dict = json.load(f)

            if json_dict is None or len(json_dict) == 0 or type(json_dict) is not dict:
                return None

        annots = []

        for obj in json_dict['features']:
            try:
                ship_data = obj['properties']
                ship_type = ship_data['ship_type']
                cls_num = 0

                poly = []
                for coord in obj['geometry']['coordinates'][0]:
                    poly.append(coord[:2])

                annot = cv2.boundingRect(np.array(poly))

                if ship_type != 'ship':
                    cls_num = 1

                if mode == 'xyxy':
                    x, y, w, h = annot
                    annot = [x, y, x+w, y+h]

                annot = list(annot)
                annot.append(cls_num)
                annots.append(annot)

            except Exception as e:
                print(f"Error : {obj}")
                continue

        return annots

    def image_aspect_ratio(self, index):
        """
        Description:
            This function is used to get the aspect ratio of the image.
        Args:
            index: The index of the image.
        Returns:
            aspect_ratio: The aspect ratio of the image.
        """

        img = cv2.imread(self.pairs[index][PAIR_IMG])
        return img.shape[1]/img.shape[0]

    @staticmethod
    def collate_fn(batch):
        """
        Description:
            This function is used to collate the batch.
        Args:
            batch: The batch of data.
        Returns:
            images: The images in the batch.
            annotations: The annotations in the batch.
        """

        images = [s['img'] for s in batch]
        annots = [s['annot'] for s in batch]

        images = torch.stack(images, dim=0)
        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.zeros((len(annots), max_num_annots, 5))
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.zeros((len(annots), 1, 5))


        return {'img': images, 'annot': annot_padded}