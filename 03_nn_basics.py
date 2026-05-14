import torch 
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    model = SimpleNN()
    print(model)

    # example input
    x = torch.rand(1, 4) # batch size of 1, 4 features
    output = model(x)
    print(output)


# Activation functions help neural networks learn complex patterns.
# Common activation functions include ReLU, Sigmoid, and Tanh.
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# Loss functions measure how well a model's predictions match the true labels.
# Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.
mse_loss = nn.MSELoss()
cross_entropy_loss = nn.CrossEntropyLoss()

# Optimizers update the model's parameters based on the computed gradients.
# Common optimizers include Stochastic Gradient Descent (SGD) and Adam.
# Adam is one of the most popular optimizers in deep learning.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # L2 regularization

# Regularization techniques like L2 regularization (weight decay) help prevent overfitting by adding a penalty to the loss function based on the magnitude of the model's parameters.

# Training Loop 
for epoch in range(10):

    # Forward pass
    outputs = model(X_train)

    # Calculate loss
    loss = criterion(outputs, y_train)

    # Reset gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# create DataLoader 
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Load Dataset 
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Building Neural Networks
# A neural network is a series of layers that process input data to make predictions.
# Common layers include:
# - Linear (fully connected) layers: perform linear transformations.
# - Convolutional layers: used for image data to capture spatial hierarchies.
# - Recurrent layers: used for sequential data like text or time series.

class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = MNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model using the training loop described above.
epochs = 5

for epoch in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# evaluate the model on the test set
model.eval() # set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Accuracy: {accuracy:.2f}%")

# save the model
torch.save(model.state_dict(), "mnist_model.pth")

# load the model
model.load_state_dict(torch.load("mnist_model.pth"))