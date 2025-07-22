import os
import json
import re
from transformers import pipeline

# Load model and tokenizer
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Prompt template
def build_prompt(description):
    return f"""You are an expert teacher.

Create one multiple-choice question suitable for a Class 1 student based on the following content:

"{description}"

- Provide four **distinct** options (A, B, C, D).
- Only one should be correct.
- Use simple language.

Ensure the format is exactly as below:
Question: <Your question here>
Options:
A. <Option 1>
B. <Option 2>
C. <Option 3>
D. <Option 4>
Answer: <Correct option letter>
"""

# Output parser
def parse_output(text, caption, source):
    try:
        print("🧾 Preprocessed Text:\n", text)

        # 1. Extract the question
        q_match = re.search(r"Question[:\-]?\s*(.*?)\n?Options[:\-]?", text, re.I | re.S)
        question = q_match.group(1).strip() if q_match else None

        # 2. Extract options (robust pattern)
        options = re.findall(r"[A-D][\.\)\s]\s*(.*?)(?=\s*[A-D][\.\)\s]|Answer:)", text, re.S)
        options = [opt.strip() for opt in options][:4]

        # 3. Extract answer
        a_match = re.search(r"Answer[:\-]?\s*([A-D])", text, re.I)
        answer = a_match.group(1).strip() if a_match else None

        if question and len(options) == 4 and answer:
            return {
                "description": caption,
                "source": source,
                "question": question,
                "options": options,
                "answer": answer
            }
        else:
            print("❌ Could not extract all required fields.")
    except Exception as e:
        print(f"❌ Exception while parsing: {e}")
    return None




final_questions = []

# ✅ Process image captions
with open(r"E:\proj_ai_test\ouput\captioned_Questions.json", "r", encoding="utf-8") as f:
    caption_data = json.load(f)

for item in caption_data:
    caption = item["caption"]
    prompt = build_prompt(caption)
    output = generator(prompt, max_new_tokens=256, do_sample=False)[0]['generated_text']
    print("🧪 Raw Output:\n", output)  # 👈 add here
    print("Generated Output (image):", output.strip())
    parsed = parse_output(output, caption, source=item["image"])
    if parsed:
        final_questions.append(parsed)
        print("✅ Parsed Question:")
        print(f"Q: {parsed['question']}")
        print(f"A. {parsed['options'][0]}")
        print(f"B. {parsed['options'][1]}")
        print(f"C. {parsed['options'][2]}")
        print(f"D. {parsed['options'][3]}")
        print(f"Answer: {parsed['answer']}\n")

    else:
        print(f"❌ Failed to parse for caption: {caption}")

# ✅ Process raw page text
with open(r"E:\proj_ai_test\ouput\output.json", "r", encoding="utf-8") as f:
    page_data = json.load(f)

for item in page_data:
    page_text = item["text"].strip()
    if not page_text.strip():
        continue

    prompt = build_prompt(page_text)
    output = generator(prompt, max_length=256, do_sample=False)[0]['generated_text']
    print("Generated Output (text):", output.strip())
    parsed = parse_output(output, page_text[:50] + "...", source=f"Page {item['page']}")
    if parsed:
        final_questions.append(parsed)
    else:
        print(f"❌ Failed to parse for page {item['page']}")



# ✅ Remove duplicates (same question and options)
unique_questions = []
seen = set()

for q in final_questions:
    q_key = (q['question'], tuple(q['options']))
    if q_key not in seen:
        seen.add(q_key)
        unique_questions.append(q)

final_questions = unique_questions

print(f"📝 Total parsed questions: {len(final_questions)}")

# ✅ Save final results
with open(r"E:\proj_ai_test\ouput\final_questions.json", "w", encoding="utf-8") as f:
    json.dump(final_questions, f, indent=2)

print("✅ Done. All valid questions saved to final_questions.json")
