import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
#import ECdata_processing as ECdata_processing
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize
from pathlib import Path

batch_size = 15
#data = ECdata_processing
root = "/data/ec259/camelyon17/raw"

# -- TRANSFORM INTO PYTORCH DATASET --

transform = Compose([
    ToTensor()
])

train_set = ImageFolder(root + '/train_17_patches', transform=transform)
valid_set = ImageFolder(root + '/validate_17_patches', transform=transform)

# Wrap an iterable over the dataset by passing to dataloader
train_dataloader = DataLoader(train_set, batch_size=batch_size)
validate_dataloader = DataLoader(valid_set, batch_size=batch_size)
print(len(validate_dataloader.dataset))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Following tutorial - https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output


model = Net().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(dataloader, model, loss_fn, optimiser):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        print(dataloader)
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n--------------------------")
    train(train_dataloader, model, loss_fn, optimiser)
    validate(validate_dataloader, model, loss_fn)
print("----- FINISHED -----")

classes = [
    "background",
    "normal",
    "tumor"
]

# model.eval()
# x, y = data.validate_dset[0][0], data.validate_dset[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
