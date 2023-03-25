import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *

combos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
allCombosAllRuns = []

for i in range(0, len(combos)):
    allRunsForCombo = []
    for j in range(0, 4):
        data = read_csv("files/combo_" + str(combos[i]) + "_" + str(j))

        valid_acc = data["valid_acc"].tolist()
        allRunsForCombo.extend(valid_acc)

    allCombosAllRuns.append(allRunsForCombo)

fig = plt.figure()
ax = plt.axes()
bp = ax.boxplot(allCombosAllRuns)
ax.set_ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], combos)
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Combination Setting Number")
plt.title("Validation Accuracies for Different Augmentation Combinations")

plt.savefig(fname="Augments_Experiment.png", dpi=350)
