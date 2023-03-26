import matplotlib.pyplot as plt
import numpy as np
import csv
from pandas import *

seeds = [0, 42, 64, 123, 256, 301, 598, 647, 999, 1011]
allSeedsAllRuns = []

for i in range(0, len(seeds)):
    accuraciesForSeed = []
    for j in range(0, 10):
        data = read_csv("seed_" + str(seeds[i]) + "_" + str(j))

        valid_acc = data["valid_acc"].tolist()
        # best = max(valid_acc)
        accuraciesForSeed.extend(valid_acc)
    allSeedsAllRuns.append(accuraciesForSeed)

fig = plt.figure()
ax = plt.axes()
bp = ax.boxplot(allSeedsAllRuns)
ax.set_ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], seeds)
plt.ylabel("Validation Accuracy Percentage (%)")
plt.xlabel("Seed Number")
plt.title("Validation Accuracies across all Epochs for Different Seeds (10 Runs)")

plt.savefig(fname="Seed_Experiment.png", dpi=350)
