### NAME: SURYA P <br>
### REG NO: 212224230280 <br> 
### Date: 11/02/2026

## EX. No. 3 : CONVOLUTIONAL DEEP NEURAL NETWORK FOR IMAGE CLASSIFICATION

## AIM :
To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset :

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Neural Network Model :

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/1b664a50-e395-4464-8f86-a5919861b4b8" />

## DESIGN STEPS :

### STEP 1 : Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).
### STEP 2 : Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3 : Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4 : Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5 : Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6 : Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7 : Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM :

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
    transforms.ToTensor(),          
    transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
```

```python
from torchsummary import summary
model = CNNClassifier()
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
print('Name: SURYA P        ')
print('Register Number: 212224230280      ')
summary(model, input_size=(1, 28, 28))
```

```python
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = model.to(device)
```

```python
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: SURYA P')
        print('Register Number: 212224230280')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model, train_loader)
```

```python
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: SURYA P')
    print('Register Number: 212224230280')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: SURYA P')
    print('Register Number: 212224230280')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: SURYA P')
    print('Register Number: 212224230280')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
test_model(model, test_loader)
```


## OUTPUT :

<img width="512" height="325" alt="image" src="https://github.com/user-attachments/assets/4b2ccb23-cb0c-4a36-ad38-7639a910554e" />

<img width="729" height="513" alt="image" src="https://github.com/user-attachments/assets/9ed16bcd-fd06-4bb7-8540-657c75edf00d" />

### Training Loss per Epoch :

<img width="409" height="220" alt="image" src="https://github.com/user-attachments/assets/3f88e374-fe16-40f7-b698-1b1474df8a4c" />

### Accuracy :

<img width="295" height="77" alt="image" src="https://github.com/user-attachments/assets/c2e0ea48-1156-461b-b393-942b7ea69c7e" />

### Confusion Matrix :

<img width="973" height="817" alt="image" src="https://github.com/user-attachments/assets/d21a1e31-cf7a-49d8-8eab-d85398e6a566" />

### Classification Report :

<img width="616" height="440" alt="image" src="https://github.com/user-attachments/assets/a768b47c-2304-4cb3-a9a6-5c32624a872e" />

### New Sample Data Prediction :

```python
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes
    print('Name: SURYA P')
    print('Register Number: 212224230280')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
predict_image(model, image_index=80, dataset=test_dataset)
```

<img width="592" height="623" alt="image" src="https://github.com/user-attachments/assets/01613196-3cb9-4957-b7d1-b9f43d40b20e" />

## RESULT :
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
