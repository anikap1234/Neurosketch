import cv2
import os

def detect_boxes(image_path):
    os.makedirs("debug", exist_ok=True)

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold for hand-drawn
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    boxes = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # 🔥 remove tiny noise
        if w * h < 20000:
            continue

        # 🔥 remove very large merged regions
        if w > 600 or h > 300:
            continue

        boxes.append({
            "id": i,
            "bbox": [x, y, x+w, y+h],
            "center": [x + w/2, y + h/2]
        })

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite("debug/stage1_boxes.png", img)

    print(f"Detected {len(boxes)} boxes")

    return img, boxes