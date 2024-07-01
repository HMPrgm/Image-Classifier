import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted to match the flattened size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = Net()

# Load the saved state dictionary
model.load_state_dict(torch.load('./models/cifar_net.pth'))

# Set the model to evaluation mode
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to display images with their predicted categories
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Load and preprocess the image
image_paths = ['./testImages/Frog.jpg','./testImages/cat.jpg']  # Add paths to your images here
for image_path in image_paths:
    input_image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)

    # Print the predicted class
    print(f'Predicted class: {classes[predicted.item()]}')

    # Display the image with the predicted category
    imshow(input_image.squeeze(), f'Predicted: {classes[predicted.item()]}')

# Example usage with CIFAR-10 test dataset
def show_test_images():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Print images
    imshow(torchvision.utils.make_grid(images), 'GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Perform inference on the batch
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Print predicted labels
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # Display the images with predicted categories
    imshow(torchvision.utils.make_grid(images), 'Predicted: ' + ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

if __name__ == '__main__':
    show_test_images()