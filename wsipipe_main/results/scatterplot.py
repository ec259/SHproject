import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *
import matplotlib.colors as mcolors

x = []
y = []
epochs = list(range(1, 33))

for i in range(0, 5):
        data = read_csv("TEST_RESULTS_" + str(i))

        valid_acc = data["valid_acc"].tolist()
        y.extend(valid_acc)
        x.extend(epochs)


fig = plt.figure()
ax = plt.axes()
bp = ax.scatter(x, y, s=70, alpha=0.3, color='tab:blue', label="Accuracy Distribution")

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(x, p(x), color='tab:orange', label="Trend Line")

ax.set_ylim(0, 100)
ax.set_xlim(1, 32)
ax.legend()
plt.yticks(np.arange(0, 101, 10))
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Number of epochs")
plt.title("Test Set Validation Accuracy (No Augmentations)")

plt.savefig(fname="TEST_SET_RESULTS.png", dpi=350)
