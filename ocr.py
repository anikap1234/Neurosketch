import pytesseract
import cv2
import json

# 🔥 TOGGLE THIS
USE_TROCR = False  # set True later after fixing torch


# -----------------------------
# OPTIONAL TrOCR (safe import)
# -----------------------------
if USE_TROCR:
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
        model.to(device)

        TROCR_AVAILABLE = True

    except Exception as e:
        print("[WARNING] TrOCR failed to load:", e)
        TROCR_AVAILABLE = False
else:
    TROCR_AVAILABLE = False


# -----------------------------
# KEYWORDS (for hybrid scoring)
# -----------------------------
KEYWORDS = ["unet", "cnn", "conv", "relu", "hog", "lstm"]


def score_text(text):
    text = text.lower()
    return sum(1 for k in KEYWORDS if k in text)


# -----------------------------
# MAIN OCR FUNCTION
# -----------------------------
def extract_texts(image, boxes):
    results = []

    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        crop = image[y1:y2, x1:x2]

        # -------- TESSERACT --------
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        t_text = pytesseract.image_to_string(
            thresh,
            config="--psm 6"
        ).lower().strip()

        final_text = t_text
        source = "tesseract"

        # -------- TrOCR (optional) --------
        if USE_TROCR and TROCR_AVAILABLE:
            try:
                pil_img = Image.fromarray(crop)

                pixel_values = processor(
                    images=pil_img,
                    return_tensors="pt"
                ).pixel_values.to(device)

                generated_ids = model.generate(pixel_values)
                trocr_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].lower().strip()

                # -------- HYBRID SELECTION --------
                if score_text(trocr_text) > score_text(t_text):
                    final_text = trocr_text
                    source = "trocr"

            except Exception as e:
                print("[WARNING] TrOCR failed on box:", e)

        print(f"[OCR:{source}] {final_text}")

        results.append({
            "id": b["id"],
            "bbox": b["bbox"],
            "center": b["center"],
            "text": final_text
        })

    # save debug
    with open("debug/stage2_ocr.json", "w") as f:
        json.dump(results, f, indent=2)

    return results