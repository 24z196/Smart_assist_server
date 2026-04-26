import cv2
import os
import re
from paddleocr import PaddleOCR
from textblob import TextBlob


# -----------------------------------
# IMAGE ENHANCEMENT (ESPCN)
# -----------------------------------
def enhance_image(image_path):
    print("Enhancing image...")

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Cannot load image")
        return image_path

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = os.path.join("weights", "ESPCN_x4.pb")

        if not os.path.exists(model_path):
            print("Warning: Model not found, skipping enhancement")
            return image_path

        sr.readModel(model_path)
        sr.setModel("espcn", 4)

        result = sr.upsample(img)

        enhanced_path = "enhanced.jpg"
        cv2.imwrite(enhanced_path, result)

        print("Super resolution completed")
        return enhanced_path

    except Exception as e:
        print(f"Enhancement failed: {e}")
        return image_path


# -----------------------------------
# TEXT CLEANING
# -----------------------------------
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".")

    try:
        text = str(TextBlob(text).correct())
    except:
        pass

    return text.capitalize()


# -----------------------------------
# OCR WITH STRUCTURE PRESERVATION
# -----------------------------------
def perform_ocr(image_path):
    print("Running PaddleOCR...")

    ocr = PaddleOCR(
        lang='en',
        drop_score=0.2,
        det_limit_side_len=1280
    )

    results = ocr.ocr(image_path)

    if not results or results[0] is None:
        print("No text detected")
        return

    boxes = results[0]

    # Sort by vertical position
    boxes = sorted(boxes, key=lambda x: x[0][0][1])

    structured_lines = []
    prev_y = None

    for box in boxes:
        raw_text = box[1][0]
        y = box[0][0][1]

        text = clean_text(raw_text)

        if prev_y is not None:
            gap = abs(y - prev_y)

            if gap > 40:
                structured_lines.append("\n\n")  # paragraph break
            elif gap > 18:
                structured_lines.append("\n")    # new line

        structured_lines.append(text)
        prev_y = y

    # Build final output
    final_text = ""
    for item in structured_lines:
        if item in ["\n", "\n\n"]:
            final_text += item
        else:
            final_text += item + " "

    print("\nFinal Extracted Text:\n")
    print(final_text.strip())


# -----------------------------------
# MAIN
# -----------------------------------
def main():
    print("OCR Pipeline Started")

    image_path = input("Enter image path: ").strip().strip('"').strip("'")

    if not image_path:
        print("No input provided")
        return

    if not os.path.exists(image_path):
        print("File not found")
        return

    enhanced_path = enhance_image(image_path)
    perform_ocr(enhanced_path)


# -----------------------------------
# ENTRY POINT
# -----------------------------------
if __name__ == "__main__":
    main()