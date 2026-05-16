import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioning:
    def __init__(self, model_name: str) -> None:
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True
        )
    def generate_caption(self, image: Image.Image):
        inputs = self.processor(images=image) 

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=40)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption