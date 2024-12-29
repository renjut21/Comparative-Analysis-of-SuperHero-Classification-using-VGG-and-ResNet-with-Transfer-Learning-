import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plt
# Define paths


train_dir = "data2/train"
val_dir = "data2/validation"
test_dir = "data2/test"




# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10
num_classes = 2




# Set up data transformations
data_transforms = {
   'train': transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
   'val': transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
    'test': transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
}


# Load datasets
image_datasets = {
   'train': datasets.ImageFolder(train_dir, data_transforms['train']),
   'val': datasets.ImageFolder(val_dir, data_transforms['val']),
   'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}


# Create dataloaders




dataloaders = {
   'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
   'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False),
   'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False),
}


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)


# Modify the classifier for binary classification
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, num_classes)


for param in model.parameters():
   param.requires_grad = False


# Unfreeze the last two layers
for param in model.classifier[-2:].parameters():
   param.requires_grad = True


model = model.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
def plot_epoch_metrics(train_losses, val_losses,test_losses, train_accs, val_accs, test_accs):


   epochs = range(1, len(train_losses) + 1)


   # Plot epoch vs loss


   plt.plot(epochs, train_losses, label='Train Loss', marker='o')
   plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
   plt.title('Epoch vs Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()
   # Plot epoch vs accuracy
   plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
   plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o')
   plt.title('Epoch vs Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()






# Training and validation loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
   """
   Train the model for one epoch.
   """
   model.train()  # Set model to training mode
   running_loss = 0.0
   running_corrects = 0


   for inputs, labels in dataloader:
       inputs, labels = inputs.to(device), labels.to(device)
       # Zero the parameter gradients
       optimizer.zero_grad()


       # Forward pass
       outputs = model(inputs)
       _, preds = torch.max(outputs, 1)
       loss = criterion(outputs, labels)


       # Backward pass and optimization
       loss.backward()
       optimizer.step()


       # Update loss and accuracy
       running_loss += loss.item() * inputs.size(0)
       running_corrects += (preds == labels).sum().item()


   epoch_loss = running_loss / len(dataloader.dataset)
   epoch_acc = running_corrects / len(dataloader.dataset)


   return epoch_loss, epoch_acc




def evaluate(model, dataloader, criterion, device):
   """
   Evaluate the model on the validation dataset.
   """
   model.eval()  # Set model to evaluation mode
   running_loss = 0.0
   running_corrects = 0


   # Turn off gradient computation for validation
   with torch.no_grad():
       for inputs, labels in dataloader:
           inputs, labels = inputs.to(device), labels.to(device)


           # Forward pass
           outputs = model(inputs)
           _, preds = torch.max(outputs, 1)
           loss = criterion(outputs, labels)


           # Update loss and accuracy
           running_loss += loss.item() * inputs.size(0)
           running_corrects += (preds == labels).sum().item()


   epoch_loss = running_loss / len(dataloader.dataset)
   epoch_acc = running_corrects / len(dataloader.dataset)


   return epoch_loss, epoch_acc




def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=1):
   """
   Train and evaluate the model for the specified number of epochs.
   """
   best_model_wts = model.state_dict()
   best_acc = 0.0
   train_losses, train_accs = [], []
   val_losses, val_accs = [], []
   test_losses, test_accs = [], []


   for epoch in range(num_epochs):
       print(f'Epoch {epoch + 1}/{num_epochs}')
       print('-' * 10)


       # Train for one epoch
       train_loss, train_acc = train_one_epoch(
           model, dataloaders['train'], criterion, optimizer, device
       )
       print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')


       # Evaluate on the validation set
       val_loss, val_acc = evaluate(
           model, dataloaders['val'], criterion, device
       )
       print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')


       test_loss, test_acc = evaluate(
           model, dataloaders['test'], criterion, device
       )
       print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
       train_losses.append(train_loss)
       train_accs.append(train_acc)
       val_losses.append(val_loss)
       val_accs.append(val_acc)
       test_losses.append(test_loss)
       test_accs.append(test_acc)


       best_model_wts = model.state_dict()


       print()
   plot_epoch_metrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs)
   # Load the best model weights
   model.load_state_dict(best_model_wts)
   print(f'Best Val Acc: {best_acc:.4f}')
   return model




# Train the model
model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)


# Save the trained model
torch.save(model.state_dict(), 'vgg16_batman_not_hero2.pth')
print("Model saved as vgg16_batman_not_hero.pth
