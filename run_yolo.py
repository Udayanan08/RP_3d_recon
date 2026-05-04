from ultralytics import YOLO
import os

# Load your YOLOv12 weights
model = YOLO("weights/best_v3.pt")

# Folder with your images
image_folder = "valid_dataset/img17.png"

# Output folder for label txt files
output_folder = "yolo_output"
os.makedirs(output_folder, exist_ok=True)

# Run inference
results = model.predict(
    source=image_folder,
    conf=0.25,
    iou=0.45,
    save=False  # we handle saving manually
)

# Save each result as YOLO format .txt
for r in results:
    image_name = os.path.splitext(os.path.basename(r.path))[0]
    label_file = os.path.join(output_folder, f"{image_name}.txt")

    with open(label_file, "w") as f:
        for box in r.boxes:
            cls = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()  # normalized values
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Saved: {label_file}")
