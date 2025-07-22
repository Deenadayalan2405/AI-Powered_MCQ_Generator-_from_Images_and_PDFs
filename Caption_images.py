# caption_images.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json

# Paths
image_dir = r"E:\proj_ai_test\ouput\images"
output_json_path = r"E:\proj_ai_test\ouput\captioned_Questions.json"

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def clean_caption(text):
    text = text.strip().lower()
    text = text.replace("marke", "mark")  # fix typo
    if len(text.split()) < 4:
        return None  # skip short captions
    return text

# Collect image captions
results = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")

        # Prompt-based captioning
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        cleaned = clean_caption(caption)
        if not cleaned:
            continue

        results.append({
            "image": f"output/images/{filename}",
            "caption": cleaned
        })

# Save to JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("✅ Captions generated and saved to captioned_questions.json")
