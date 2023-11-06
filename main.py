import os
import torch

from data_loading import create_data_loader
from src.model import load_model


dataset_folders = [
    'squirrels_head',
    'squirrels_tail',
    'the_center_of_the_gemstone',
    'the_center_of_the_koalas_nose',
    'the_center_of_the_owls_head',
    'the_center_of_the_seahorses_head',
    'the_center_of_the_teddy_bear_nose'
]


def evaluate_dataset():
    global model
    ds_path = os.path.join("..", "tasks", dataset_folders[2])
    data_loader, n_samples = create_data_loader(ds_path, load_images=True, batch_size=2)

    true_positives = 0
    distances = 0

    for iteration, data in enumerate(data_loader):
        images, targets = data
        boxes_centers = model(images, texts=[["gemstone"] for i in range(targets.shape[0])])

        batch_distances = torch.norm(boxes_centers - targets, dim=1)
        batch_true_positives = torch.sum(batch_distances < 0.1)

        true_positives += batch_true_positives.item()
        distances += torch.sum(batch_distances)

        if iteration == 10:
            break

    accuracy = true_positives / n_samples
    avg_distances = distances / n_samples


if __name__ == "__main__":
    model = load_model()
    evaluate_dataset()
