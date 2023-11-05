import os
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from json.decoder import JSONDecodeError


dataset_folders = [
    'squirrels_head',
    'squirrels_tail',
    'the_center_of_the_gemstone',
    'the_center_of_the_koalas_nose',
    'the_center_of_the_owls_head',
    'the_center_of_the_seahorses_head',
    'the_center_of_the_teddy_bear_nose'
]


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
        for image_name in tqdm(all_files):
            json_name = os.path.splitext(image_name)[0] + '.json'
            json_path = os.path.join(self.data_folder, json_name)
            if os.path.exists(json_path):
                with open(json_path, "r") as json_file:
                    try:
                        coords = json.loads(json_file.read())
                    except JSONDecodeError:
                        empty_json_files += 1
                        continue

                    if coords:
                        self.coordinates.append((coords[0]['x'], coords[0]['y']))
                    else:
                        continue
            else:
                print(f"There is no json file for {image_name}")
                continue

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
        return image, self.coordinates[idx]


def create_data_loader(folder_path: str, load_images: bool, batch_size: int) -> DataLoader:
    dataset = TasksDataset(folder_path, load_images=load_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
