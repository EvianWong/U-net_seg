import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Contracting path
        self.conv1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.conv5 = DoubleConv(512, 1024)
        
        # Expanding path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        
        # Output layer
        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        conv1_out = self.conv1(x)
        pool1_out = self.pool1(conv1_out)
        conv2_out = self.conv2(pool1_out)
        pool2_out = self.pool2(conv2_out)
        conv3_out = self.conv3(pool2_out)
        pool3_out = self.pool3(conv3_out)
        conv4_out = self.conv4(pool3_out)
        pool4_out = self.pool4(conv4_out)
        
        # Bottleneck
        conv5_out = self.conv5(pool4_out)
        
        # Expanding path
        upconv4_out = self.upconv4(conv5_out)
        concat4 = torch.cat((conv4_out, upconv4_out), dim=1)
        conv6_out = self.conv6(concat4)
        
        upconv3_out = self.upconv3(conv6_out)
        concat3 = torch.cat((conv3_out, upconv3_out), dim=1)
        conv7_out = self.conv7(concat3)
        
        upconv2_out = self.upconv2(conv7_out)
        concat2 = torch.cat((conv2_out, upconv2_out), dim=1)
        conv8_out = self.conv8(concat2)
        
        upconv1_out = self.upconv1(conv8_out)
        concat1 = torch.cat((conv1_out, upconv1_out), dim=1)
        conv9_out = self.conv9(concat1)
        
        # Output layer
        output = self.conv10(conv9_out)
        output = torch.sigmoid(output)
        
        return output


# Custom dataset class for training and testing
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Load and preprocess your dataset here
        pass

    def __len__(self):
        # Return the total number of samples in the dataset
        pass

    def __getitem__(self, idx):
        # Return a single sample from the dataset
        pass


# Training loop
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        # Move input and target tensors to the device (e.g. GPU)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# Testing loop
def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move input and target tensors to the device (e.g. GPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(dataloader)


# Set device (CPU or GPU) for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the U-Net model
model = UNet().to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create training and testing datasets and dataloaders
train_dataset = CustomDataset()
test_dataset = CustomDataset()

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training and testing loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer)
    test_loss = test(model, test_dataloader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")
