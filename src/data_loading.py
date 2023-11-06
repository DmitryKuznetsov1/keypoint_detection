import os
from typing import Any, Tuple

from PIL.JpegImagePlugin import JpegImageFile
from PIL import Image
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from json.decoder import JSONDecodeError
import numpy as np


# TODO: Добавить обработку нескольких точек из json
class TasksDataset(Dataset):
    def __init__(self, data_folder: str, load_images=True):
        self.data_folder = data_folder
        self.load_images = load_images

        self.images = []
        self.coordinates = []
        all_files = (f for f in os.listdir(data_folder) if f.endswith('.jpg'))

        print(f"Loading {data_folder} ...")
        empty_json_files = 0
        for image_name in tqdm(all_files, dynamic_ncols=True):
            json_name = os.path.splitext(image_name)[0] + '.json'
            json_path = os.path.join(self.data_folder, json_name)

            if not os.path.exists(json_path):
                print(f"There is no json file for {image_name}")
                continue

            with open(json_path, "r") as json_file:
                try:
                    coords = json.loads(json_file.read())
                except JSONDecodeError:
                    empty_json_files += 1
                    continue
                if not coords:
                    continue

                self.coordinates.append((coords[0]['x'], coords[0]['y']))

            image_path = os.path.join(self.data_folder, image_name)
            if self.load_images:
                image = Image.open(image_path)
                self.images.append(image)
            else:
                self.images.append(image_path)

        print(f"Completed with {empty_json_files} empty JSON files")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.load_images:
            image = self.images[idx]
        else:
            image_path = self.images[idx]
            image = Image.open(image_path)
        return self.transform(image), torch.Tensor(self.coordinates[idx])

    @staticmethod
    def transform(image: JpegImageFile):
        image_array = np.array(image)
        return image_array


def create_data_loader(folder_path: str, load_images: bool, batch_size: int) -> Tuple[DataLoader[Any], int]:
    dataset = TasksDataset(folder_path, load_images=load_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, len(dataset)
