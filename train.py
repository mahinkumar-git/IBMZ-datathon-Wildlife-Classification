import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Selecting the Dataset Directory
data_dir = "dataset"

# The dataset consist of 350 Different Species of Animals and Birds
# And about 16000 Images overall neatly classified with names
train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
num_classes = 350

# We import a pretrained Model - Resnet 152
# This provides as advantage of transfer learning
# Keeps the model Much Smaller in size than regular CNN Models trained from scratch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

# Setting Up to use a CUDA Device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Setting up Optimizer and Criterion
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Function 
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Starting to train the model
        # Setting up stats counters
        model.train() 
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        start_time = time.time()

        # We Iterate over data in batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass for the model
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization for the model
            loss.backward()
            optimizer.step()

            # Update stat for the training progress
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            # Print progress every 10 batches
            if total_samples % 10 == 0:
                batch_loss = running_loss / total_samples
                batch_acc = running_corrects.double() / total_samples
                print(f"Progress: {total_samples}/{len(train_dataset)} - Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")

        # End of epoch stats at each epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Time for epoch {epoch+1}: {time.time() - start_time:.2f} seconds\n')

    # we return the trained model
    return model


# Model Training
model = train_model(model, criterion, optimizer, num_epochs=15)

# Saving the Trained Model
torch.save(model.state_dict(),"Wildlife_model.pth")
