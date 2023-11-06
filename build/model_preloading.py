from transformers import Owlv2Processor, Owlv2ForObjectDetection

model_weights = "google/owlv2-base-patch16-ensemble"
processor = Owlv2Processor.from_pretrained(model_weights)
model = Owlv2ForObjectDetection.from_pretrained(model_weights)

print("Successfully loaded google/owlv2-base-patch16-ensemble")
