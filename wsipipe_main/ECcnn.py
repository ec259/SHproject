import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import ECdata_processing as ECdata_processing


batch_size = 64

# Wrap an iterable over the dataset by passing to dataloader
train_dataloader = DataLoader(ECdata_processing.train_dset, batch_size=batch_size)
validate_dataloader = DataLoader(ECdata_processing.validate_dset, batch_size=batch_size)

for X, y in validate_dataloader:
    # N - number of images (in the batch), C - Number of channels(?) , H & W - height and width images are 28x28.
    print(X)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __int__(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLu(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # Backpropogation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
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
print("Done!")

classes = [
    "background",
    "normal",
    "tumor"
]

model.eval()
x, y = ECdata_processing.validate_dset[0][0], ECdata_processing.validate_dset[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
