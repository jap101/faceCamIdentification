import torch
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image, ImageDraw
from torchvision.transforms import transforms


# Load the pre-trained model
#model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = models.resnet101(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('object_detection_model.pth'))
model.eval()

# Move the model to the appropriate device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set the directory for the test images
test_image_dir = 'dataset/unseen_test/output_frames'

# Loop through the test images and make predictions
for filename in os.listdir(test_image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(test_image_dir, filename)
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)

        # Make the prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_box = output.squeeze().tolist()
            
        x1, y1, x2, y2 = predicted_box
        x1 = int(x1 * original_width / 224)
        y1 = int(y1 * original_height / 224)
        x2 = int(x2 * original_width / 224)
        y2 = int(y2 * original_height / 224)
        predicted_box = [x1, y1, x2, y2]
        
        # Draw the bounding box on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle(predicted_box, outline=(255, 0, 0), width=2)

        # Save the image with the bounding box
        output_path = os.path.join('dataset/unseen_test/model_output', filename)
        image.save(output_path)
        print(f'Saved image with predicted bounding box: {output_path}')