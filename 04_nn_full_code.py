import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


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


# Load model
model = MNISTModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load image
image = Image.open("my_digit.png")

# Transform image
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)


# Prediction
with torch.no_grad():
    outputs = model(image_tensor)
    predicted_class = torch.argmax(outputs, dim=1)


print(f"Predicted Digit: {predicted_class.item()}")


# Show image
plt.imshow(image, cmap="gray")
plt.title(f"Predicted Digit: {predicted_class.item()}")
plt.axis("off")
plt.show()