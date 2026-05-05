import sys
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioning:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            use_safetensors=True
        )
        

    def generate_caption(self, input_image) -> str:


        raw_image = Image.open(input_image).convert("RGB")

        inputs = self.processor(images=raw_image, return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=40)

        description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
        return description