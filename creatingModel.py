import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
import os
import torchvision.ops as ops

import torch.nn.functional as F
from tqdm import tqdm


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate the true positives (TP), false positives (FP), and false negatives (FN)
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        # Calculate the Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal Tversky Loss
        loss = 1 - torch.pow(tversky_index, self.gamma)

        return loss

# Custom dataset class
class ImageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        image_name = self.data_df.iloc[index, 0]
        image = Image.open(os.path.join('dataset/output_frames', image_name))

        # Extract the bounding box coordinates from the data_df
        x1 = self.data_df.iloc[index, 5]
        y1 = self.data_df.iloc[index, 6]
        x2 = self.data_df.iloc[index, 7]
        y2 = self.data_df.iloc[index, 8]

        # Your label processing logic
        label = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Load the data
data = pd.read_csv('data.csv')
train_data = data.sample(frac=0.7, random_state=42)
val_data = data.loc[~data.index.isin(train_data.index)]

# Preprocess the data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Create the datasets and dataloaders
train_dataset = ImageDetectionDataset(train_data, transform=transform)
val_dataset = ImageDetectionDataset(val_data, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the pre-trained model
#model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#model = models.resnet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.resnet101(pretrained=True)

# Freeze the base layers
for param in model.parameters():
    param.requires_grad = False

# Add custom layers
model.fc = nn.Linear(model.fc.in_features, 4)  # Output 4 values for the bounding box coordinates
model = model.to(torch.float32)  # Convert the model to float32

# Train the model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
#criterion = nn.MSELoss()
criterion = FocalTverskyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 10





# Training loop
for epoch in range(num_epochs):
    # Training loop
    train_loss = 0.0
    model.train()
    with tqdm(train_dataloader, unit="batch") as train_progress_bar:
        train_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_progress_bar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_progress_bar.set_postfix(loss=loss.item())
            train_loss += loss.item()
    train_loss /= len(train_dataloader)

    # Validation loop
    val_loss = 0.0
    model.eval()
    with tqdm(val_dataloader, unit="batch") as val_progress_bar:
        val_progress_bar.set_description(f"Validation")
        for images, labels in val_progress_bar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_progress_bar.set_postfix(loss=loss.item())
            val_loss += loss.item()
    val_loss /= len(val_dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

'''
for epoch in range(num_epochs):
    # Training loop
    for images, labels in train_dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            outputs = model(images)

            # Compute the IoU between the predicted and ground truth bounding boxes
            #predicted_boxes = outputs
            #ground_truth_boxes = labels
            #iou = ops.box_iou(predicted_boxes, ground_truth_boxes)

            loss = criterion(outputs, labels)


            # Compute other validation metrics like precision, recall, F1-score, etc.
            # and update the corresponding variables

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
'''



torch.save(model.state_dict(), 'object_detection_model.pth')
