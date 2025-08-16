import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


# =========================
# install dependencies
# 1.pip3 install pymupdf      
# 2.pip install pytesseract  
# 3.Download the Windows installer https://github.com/UB-Mannheim/tesseract/wiki
# to run : LexBudget\scripts>python text-image-extraction.py
# =========================

# =========================
# CONFIGURATION
# =========================

pdf_path = "text-image-test.pdf"   # path to your PDF file

# If Tesseract is not in PATH, set full path to executable, e.g.:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#  Print the Tesseract version
print(pytesseract.get_tesseract_version())

# =========================
# EXTRACT TEXT
# =========================

doc = fitz.open(pdf_path)
all_text = ""

for page_num in range(len(doc)):
    page = doc[page_num]

    # Get images on the page
    images = page.get_images(full=True)
    if not images:
        continue  # skip if no images

    for img_index, img in enumerate(images):
        xref = img[0]  # image reference
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        # Convert to PIL image
        img_pil = Image.open(io.BytesIO(image_bytes))

        # Run OCR
        text = pytesseract.image_to_string(img_pil)
        all_text += f"\n--- Page {page_num + 1}, Image {img_index + 1} ---\n{text}"

doc.close()

# Print results
print("Extracted Text:")
print(all_text)

# Optionally save to file
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
