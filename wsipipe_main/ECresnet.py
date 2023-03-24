import csv
import time
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
# import ECdata_processing as ECdata_processing
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

from ECaugmentation_transforms import *


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def run(seed, filename, aug_combo, lr, n_epochs):
    root = "/data/ec259/camelyon17/raw"
    torch.manual_seed(seed)
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    batch_size = 16

    # Augmentation experiments
    if aug_combo is None:
        train_set = ImageFolder(root + '/patches/train_patches', transform=preprocess)
        valid_set = ImageFolder(root + '/patches/train_patches', transform=preprocess)
        train_augmentations_set = train_set
    else:
        train_set = ImageFolder(root + '/patches/train_patches', transform=preprocess)
        # TEST SET for final experiment
        valid_set = ImageFolder(root + '/patches/test_patches', transform=preprocess)
        augmented_set = ImageFolder(root + '/patches/train_patches', transform=aug_combo)
        train_augmentations_set = ConcatDataset([train_set, augmented_set])

    # Wrap an iterable over the dataset by passing to dataloader
    train_dataloader = DataLoader(train_augmentations_set, batch_size=batch_size)
    validate_dataloader = DataLoader(valid_set, batch_size=batch_size)

    # Get cpu or gpu device for training
    net = resnet50(weights=weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.cuda() if device else net
    net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, 2)
    net.fc = net.fc.cuda() if torch.cuda.is_available() else net.fc

    print_every = 100
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)

    start = time.time()

    # https://www.pluralsight.com/guides/introduction-to-resnet
    # START OF CITED CODE
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')

        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)

        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0

        with torch.no_grad():
            net.eval()
            for data_t, target_t in (validate_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)

            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(validate_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')

        net.train()
    # END OF CITED CODE

    print(f"Finished Training -- Time taken: {time.time() - start} seconds")
    print(f"Train loss: {train_loss}")
    print(f"Train Accuracy: {train_acc}")

    file = open(filename, "w")
    writer = csv.writer(file)
    writer.writerow(["train_loss", "train_acc", "valid_loss", "valid_acc"])
    for w in range(len(train_acc)):
        writer.writerow([train_loss[w], train_acc[w], val_loss[w], val_acc[w]])
    file.close()


