import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 14})
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
for i in range(0, len(learning_rates)):
    data = read_csv(str(learning_rates[i]))
    train_acc = data["train_acc"].tolist()
    valid_acc = data["valid_acc"].tolist()
    epochs = list(range(1, 11))

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(epochs, train_acc, color='tab:orange', label="Training Accuracy")
    plt.plot(epochs, valid_acc, color='tab:blue', label="Testing Accuracy")

    ax.set_ylim(0, 100)
    ax.set_xlim(1, 10)
    ax.legend()
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel("Accuracy Percentage (%)")
    plt.xlabel("Number of epochs")
    plt.title("Train vs Validation Accuracy for LR=" + str(learning_rates[i]))

    plt.savefig(fname="LR_" + str(learning_rates[i]) + ".png", dpi=350)
