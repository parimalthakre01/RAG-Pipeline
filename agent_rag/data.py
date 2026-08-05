from pypdf import PdfReader
import os

pdf_dir = os.path.join("..", "data")
output_dir = os.path.join("..", "text_data")
os.makedirs(output_dir, exist_ok=True)

def create_pdf_text():
    text = ""
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, filename))
            text = "".join(page.extract_text() for page in reader.pages)
            
            txt_filename = filename.replace(".pdf", ".txt")
            with open(os.path.join(output_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"Saved: {txt_filename}")
            documents.append({"filename": filename, "text": text})

if __name__ == "__main__":
    create_pdf_text()