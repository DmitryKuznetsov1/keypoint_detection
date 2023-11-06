from typing import Iterable, List

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def select_boxes_with_highest_scores(results: Iterable):
    most_relevant_boxes = torch.stack([r['boxes'][torch.argmax(r['scores'])] for r in results])
    return most_relevant_boxes


def get_relative_center_coordinates(boxes: torch.Tensor, image_height: int, image_width: int):
    y_c = (boxes[:, 0] + boxes[:, 2]) / 2 / image_height
    x_c = (boxes[:, 1] + boxes[:, 3]) / 2 / image_width
    p_c = torch.stack((y_c, x_c)).T
    return p_c


def load_model(weights: str = "google/owlv2-base-patch16-ensemble", device: torch.device = torch.device('cpu')):
    """
    Args:
        weights: The name of the pretrained model that is loaded from the Hugging Face Hub
        device: cuda device or cpu
    Returns:
        function: A function that takes an image and text queries and returns a torch.Tensor of predicted centers.
    """
    processor = Owlv2Processor.from_pretrained(weights)
    model = Owlv2ForObjectDetection.from_pretrained(weights)
    model.eval()
    model.to(device)

    def predict(images: torch.Tensor, texts: List[list]):
        with torch.no_grad():
            images = images.to(device)
            inputs = processor(text=texts, images=images, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.Tensor([image.size()[:-1] for image in images])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)
            boxes = select_boxes_with_highest_scores(results)
            boxes_centers = get_relative_center_coordinates(boxes, images.shape[1], images.shape[2])
        return boxes_centers

    return predict
