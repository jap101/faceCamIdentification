import torch
import torch.nn as nn
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO
import torchvision.ops as ops

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

class GIoULoss(nn.Module):
    def __init__(self):
        super(GIoULoss, self).__init__()

    def forward(self, pred, target):
        giou = ops.generalized_box_iou(pred, target)
        loss = 1 - giou
        return loss.mean()

class ImageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.data_df.iloc[index, 0]
        image = Image.open(os.path.join('dataset/output_frames', image_name))

        # Extract the bounding box coordinates from the data_df
        x1 = self.data_df.iloc[index, 5]
        y1 = self.data_df.iloc[index, 6]
        x2 = self.data_df.iloc[index, 7]
        y2 = self.data_df.iloc[index, 8]

        # Convert the bounding box coordinates to the format expected by YOLO
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        label = torch.tensor([x, y, width, height], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_df)

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
model = YOLO('yolov5s.pt')

# Freeze the backbone layers
freeze = [f'model.{x}.' for x in range(12)]
for k, v in model.named_parameters():
    v.requires_grad = True
    if any(x in k for x in freeze):
        print(f'freezing {k}')
        v.requires_grad = False

# Add a custom head on top of the feature extractor
model.model.head[-1] = nn.Sequential(
    nn.Linear(model.model.head[-1][0].in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4)  # Output 4 values for the bounding box
)

# Train the model
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = GIoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
num_epochs = 50

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        results = model(images)
        loss = results[0].loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            results = model(images)
            val_loss += results[0].loss.item()
    val_loss /= len(val_dataloader)

    # Update the learning rate
    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model.state_dict(), 'object_detection_model.pth')




'''import torch
import torch.nn as nn
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO

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
    
    
import torchvision.ops as ops

class GIoULoss(nn.Module):
    def __init__(self):
        super(GIoULoss, self).__init__()

    def forward(self, pred, target):
        giou = ops.generalized_box_iou(pred, target)
        loss = 1 - giou
        return loss.mean()

class ImageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.data_df.iloc[index, 0]
        image = Image.open(os.path.join('dataset/output_frames', image_name))

        # Extract the bounding box coordinates from the data_df
        x1 = self.data_df.iloc[index, 5]
        y1 = self.data_df.iloc[index, 6]
        x2 = self.data_df.iloc[index, 7]
        y2 = self.data_df.iloc[index, 8]

        # Convert the bounding box coordinates to the format expected by YOLO
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        label = torch.tensor([x, y, width, height], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_df)

class CustomYOLO(nn.Module):
    def __init__(self, model_path, num_classes=4):
        super().__init__()
        self.yolo_model = YOLO(model_path).model

        # Freeze the feature extractor layers
        for param in self.yolo_model.parameters():
            param.requires_grad = False

        # Add a custom head on top of the feature extractor
        self.head = nn.Sequential(
            nn.Linear(self.yolo_model.fc[-1].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.yolo_model(x)
        output = self.head(features.squeeze())
        return output

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
model = CustomYOLO('yolov5s.pt')
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train the model
criterion = GIoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
num_epochs = 50

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        results = model(images)
        loss = results[0].loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            results = model(images)
            val_loss += results[0].loss.item()
    val_loss /= len(val_dataloader)

    # Update the learning rate
    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
torch.save(model.state_dict(), 'object_detection_model.pth')







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
model = YOLO('yolov5s.pt')

# Freeze the feature extractor layers
for param in model.model.parameters():
    param.requires_grad = False


# Add a custom head on top of the feature extractor
model.model.head[-1] = nn.Sequential(
    nn.Linear(model.model.head[-1][0].in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4)  # Output 4 values for the bounding box
)

# Train the model
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = GIoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
num_epochs = 50

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        results = model(images)
        loss = results[0].loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            results = model(images)
            val_loss += results[0].loss.item()
    val_loss /= len(val_dataloader)

    # Update the learning rate
    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model.state_dict(), 'object_detection_model.pth')
    
    '''