import os
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')

# Set the device (CPU, GPU, or MPS)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the directory containing the unseen data
unseen_data_dir = 'dataset/unseen_test/output_frames'

# Create a directory to save the results
results_dir = 'dataset/unseen_test/model_output'
os.makedirs(results_dir, exist_ok=True)

# Iterate through the unseen data
for filename in os.listdir(unseen_data_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(unseen_data_dir, filename)
        image = Image.open(image_path)

        # Make a prediction using the YOLO model
        results = model(image_path)

        # Get the bounding box with the highest confidence
        if results[0].boxes:
            max_conf_box = results[0].boxes[results[0].boxes.conf.argmax()]
            x1, y1, x2, y2 = max_conf_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box on the image
            image_with_boxes = image.copy()
            draw = ImageDraw.Draw(image_with_boxes)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)

            # Save the image with the bounding box
            output_path = os.path.join(results_dir, filename)
            image_with_boxes.save(output_path)
        else:
            # Save the original image if no bounding box is detected
            output_path = os.path.join(results_dir, filename)
            image.save(output_path)

print("Results saved to:", results_dir)