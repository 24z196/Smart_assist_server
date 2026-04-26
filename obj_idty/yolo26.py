from ultralytics import YOLO
import cv2
import os


def detect_objects(image_path):
    # Check file
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Cannot read image.")
        return

    # Load YOLO model (pretrained)
    model = YOLO("yolo26x.pt")   

    # Run detection
    results = model(img)

    print("\nDetected Objects:\n")

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[cls_id]

            print(f"{label} ({confidence:.2f})")

    # Show image with bounding boxes
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("OBJECT DETECTION (YOLO)\n")

    image_path = input("Enter image path: ").strip().strip('"').strip("'")

    detect_objects(image_path)