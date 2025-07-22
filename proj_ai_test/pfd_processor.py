import fitz  # PyMuPDF
import os
import json

# Set your PDF path
pdf_path = "E:/proj_ai_test/input.pdf"
output_image_dir = "E:\proj_ai_test\ouput\images"
output_json_path = "E:\proj_ai_test\ouput/output.json"

# Make sure output directory exists
os.makedirs(output_image_dir, exist_ok=True)

# Open PDF
doc = fitz.open(pdf_path)

output = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text().strip()
    images = page.get_images(full=True)

    image_paths = []

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        img_bytes = base_image["image"]
        ext = base_image["ext"]
        img_filename = f"page{page_num+1}_image{img_index+1}.{ext}"
        img_path = os.path.join(output_image_dir, img_filename)
        
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        
        image_paths.append(img_path)

    output.append({
        "page": page_num + 1,
        "text": text,
        "images": image_paths
    })

# Save to JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("✅ Extraction completed.")
print(f"Images saved to: {output_image_dir}")
print(f"JSON saved to: {output_json_path}")
