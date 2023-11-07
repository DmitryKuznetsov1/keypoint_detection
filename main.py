import argparse
import os
from time import time
from typing import Tuple

from tqdm import tqdm

import torch

from src.data_loading import create_data_loader
from src.model import load_model
from torch.utils.data import DataLoader


dataset = {
    'squirrels_head': 'head',
    'squirrels_tail': 'tail',
    'the_center_of_the_gemstone': 'gemstone',
    'the_center_of_the_koalas_nose': 'nose',
    'the_center_of_the_owls_head': 'head',
    'the_center_of_the_seahorses_head': 'head',
    'the_center_of_the_teddy_bear_nose': 'nose',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-path", type=str, default="tasks",
                        help="Path to the folder containing the specified datasets")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--load-to-memory", dest="load_images", type=int, default=1, help="Load datasets into RAM")
    args = parser.parse_args()
    args.load_images = False if args.load_images == 0 else True

    print(f"Batch size: {args.batch_size}")
    print(f"Load to memory: {args.load_images}\n")
    return args


def evaluate_model(model, data_loader: DataLoader, query: str) -> Tuple[int, float]:
    true_positives = 0
    distances = 0

    for iteration, data in tqdm(enumerate(data_loader)):
        images, targets = data
        boxes_centers = model(images, texts=[[query] for i in range(targets.shape[0])])

        batch_distances = torch.norm(boxes_centers - targets, dim=1)
        batch_true_positives = torch.sum(batch_distances < 0.1)

        true_positives += batch_true_positives.item()
        distances += torch.sum(batch_distances).item()

    return true_positives, distances


def main():
    args = parse_args()
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)

    true_positives_total = 0
    distances_total = 0
    n_samples_total = 0
    untracked_points_total = 0

    start = time()
    for dataset_folder_name, dataset_query in dataset.items():
        ds_start = time()
        ds_path = os.path.join(args.datasets_path, dataset_folder_name)
        data_loader, n_samples, untracked_points = create_data_loader(ds_path, load_images=args.load_images, batch_size=args.batch_size)
        ds_true_positives, ds_distances = evaluate_model(model, data_loader, dataset_query)

        ds_accuracy = ds_true_positives / (n_samples + untracked_points)
        ds_avg_distance = ds_distances / n_samples
        print(f"\tacc: {ds_accuracy:.2f}, dst: {ds_avg_distance:.3f}, time: {(time() - ds_start):.2f}\n")

        true_positives_total += ds_true_positives
        distances_total += ds_distances
        n_samples_total += n_samples
        untracked_points_total += untracked_points

    accuracy = true_positives_total / (n_samples_total + untracked_points_total)
    avg_distance = distances_total / n_samples_total
    print(f"All Datasets: Accuracy: {accuracy}, Average Distance: {avg_distance}, Time: {round(time() - start, 2)}")


if __name__ == "__main__":
    main()
