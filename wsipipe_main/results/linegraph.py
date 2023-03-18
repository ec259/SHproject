import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *
import matplotlib.colors as mcolors


data = read_csv("TEST_RESULTS_2")
train_acc = data["train_acc"].toList()
valid_acc = data["valid_acc"].tolist()
epochs = list(range(1,33))

fig = plt.figure()
ax = plt.axes()
plt.plot(epochs, train_acc, color='tab:orange', label="Training Accuracy")
plt.plot(epochs, valid_acc, color='tab:blue', label="Testing Accuracy")

ax.set_ylim(0, 100)
ax.set_xlim(1, 32)
ax.legend()
plt.yticks(np.arange(0, 101, 10))
plt.ylabel("Accuracy Percentage (%)")
plt.xlabel("Number of epochs")
plt.title("Train vs Test Accuracy")

plt.savefig(fname="TEST_SET_2_TrainVsTest.png", dpi=350)
