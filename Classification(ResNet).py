import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from matplotlib import pyplot as plt
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define data transformations
transform = transforms.Compose([
   transforms.Resize((224, 224)),  # Resize to fit ResNet input
   transforms.ToTensor(),         # Convert to Tensor
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pre-trained ResNet
])


# Load datasets
dataset = datasets.ImageFolder('data2/train', transform=transform)
train_size = int(0.6 * len(dataset))
val_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Load pre-trained ResNet
model = models.resnet18(pretrained=True)




for param in model.parameters():
   param.requires_grad = False


# Unfreeze the last residual block and the fully connected layer
for param in model.layer4.parameters():  # Unfreezing the last residual block
   param.requires_grad = True
for param in model.fc.parameters():     # Unfreezing the FC layer
   param.requires_grad = True


model.fc = nn.Linear(model.fc.in_features, 2)
# Add a new output layer on top of the existing fc layer
# class ModifiedResNet(nn.Module):
#     def __init__(self, base_model):
#         super(ModifiedResNet, self).__init__()
#         self.base_model = base_model
#         self.new_fc = nn.Linear(base_model.fc.out_features, 2)  # 2 output classes: "Batman" and "Not a Hero"
#     def forward(self, x):
#         x = self.base_model(x)
#         x = self.new_fc(x)
#         return x


# Wrap the original model with the modified architecture
model = model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training and validation
num_epochs = 10
train_losses = []
validation_losses = []
accuracies = []
epochs = []
for epoch in range(num_epochs):
   # Training
   model.train()
   running_loss = 0.0
   val_size = int(0.2 * len(dataset))
   train_size = int(0.8 * len(dataset))


   for images, labels in train_loader:
       images, labels = images.to(device), labels.to(device)


       # Forward pass
       outputs = F.softmax(model(images))
       loss = criterion(outputs, labels)


       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


       running_loss += loss.item()
   train_loss = running_loss / len(train_loader)


   # Validation
   model.eval()
   correct = 0
   total = 0
   validation_loss = 0.0
   with torch.no_grad():
       for images, labels in val_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           loss = criterion(outputs, labels)
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
           validation_loss += loss.item()
   val_accuracy = correct / total
   epochs.append(epoch)
   validation_losses.append(validation_loss)
   train_losses.append(train_loss)
   accuracies.append(val_accuracy)
   print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {validation_loss:.4f}")


# Testing
plt.plot(epochs, train_losses, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.plot(epochs, validation_losses, label="Validation Loss")
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.plot(epochs, accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


model.eval()
correct = 0
total = 0
with torch.no_grad():
   for images, labels in test_loader:
       images, labels = images.to(device), labels.to(device)
       outputs = model(images)
       _, predicted = torch.max(outputs, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")


# Save the final model weights
torch.save(model.state_dict(), 'final_resnet_model2.pth')
