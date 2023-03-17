import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *

seeds = [0, 42, 64, 123, 256, 301, 598, 647, 999, 1011]
allSeedsAllRuns = []

for i in range(0, len(seeds)):
    allRunsForSeed = []
    for j in range(0, 10):
        data = read_csv("./seeds/seed_" + str(seeds[i]) + "_" + str(j))

        valid_acc = data["valid_acc"].tolist()
        allRunsForSeed.extend(valid_acc)

    allSeedsAllRuns.append(allRunsForSeed)

fig = plt.figure()
ax = plt.axes()
bp = ax.boxplot(allSeedsAllRuns)
ax.set_ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], seeds)
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Seed Number")
plt.title("Validation Accuracies for Different Seeds")

plt.savefig(fname="Seed_Experiment.png", dpi=350)

# AUGMENTATIONS
combos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
allCombosAllRuns = []

for i in range(0, len(combos)):
    allRunsForCombo = []
    for j in range(0, 4):
        data = read_csv("./augments_experiment/combo_" + str(combos[i]) + "_" + str(j))

        valid_acc = data["valid_acc"].tolist()
        allRunsForCombo.extend(valid_acc)

    allCombosAllRuns.append(allRunsForCombo)

fig = plt.figure()
ax = plt.axes()
bp = ax.boxplot(allCombosAllRuns)
ax.set_ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], combos)
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Combination Setting Number")
plt.title("Validation Accuracies for Different Augmentation Combinations")

plt.savefig(fname="Augments_Experiment.png", dpi=350)
