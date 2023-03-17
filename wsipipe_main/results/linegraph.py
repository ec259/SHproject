import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *

x = []
y = []
epochs = list(range(1,32))

for i in range(0, 5):
        data = read_csv("./results/TEST_RESULTS_" + str(i))

        valid_acc = data["valid_acc"].tolist()
        y.extend(valid_acc)
        x.extend(epochs)


fig = plt.figure()
ax = plt.axes()
bp = ax.plot(x, y)
ax.set_ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Number of epochs")
plt.title("Test Set Validation Accuracy (No Augmentations)")

plt.savefig(fname="TEST_SET_RESULTS.png", dpi=350)
