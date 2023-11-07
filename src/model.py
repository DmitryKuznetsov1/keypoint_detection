from typing import Iterable, List

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def select_boxes_with_highest_scores(results: Iterable) -> torch.Tensor:
    most_relevant_boxes = []
    for r in results:
        if r['scores'].numel() > 0:
            most_relevant_boxes.append(r['boxes'][torch.argmax(r['scores'])])
        else:
            most_relevant_boxes.append(torch.Tensor([0., 0., 0., 0., ]))
    return torch.stack(most_relevant_boxes)


def get_relative_center_coordinates(boxes: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    y_c = (boxes[:, 0] + boxes[:, 2]) / 2 / image_height
    x_c = (boxes[:, 1] + boxes[:, 3]) / 2 / image_width
    p_c = torch.stack((y_c, x_c)).T
    return p_c


def outputs_to_cpu(outputs):
    for k in outputs.keys():
        if isinstance(outputs[k], torch.Tensor):
            outputs[k] = outputs[k].cpu()
    return outputs


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
    model = model.to(device)
    model.eval()

    def predict(images: torch.Tensor, texts: List[list], device=device):
        with torch.no_grad():
            images = images.to(device)
            inputs = processor(text=texts, images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            outputs = outputs_to_cpu(outputs)
            target_sizes = torch.Tensor([image.size()[:-1] for image in images])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)
            boxes = select_boxes_with_highest_scores(results)
            boxes_centers = get_relative_center_coordinates(boxes, images.shape[1], images.shape[2])
        return boxes_centers

    return predict
