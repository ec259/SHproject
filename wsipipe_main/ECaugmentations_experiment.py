from ECresnet import run as model
from ECaugmentation_transforms import *

seed = 99
augments = [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8, combo_9, combo_10, combo_11,
            combo_12, combo_13, combo_14, combo_15]

# for i in range(0, 16):
#     for j in range(0, 5):
#         filename = f"./results/augments_experiment/combo_{i+1}_{j}"
#         model(seed=seed, filename=filename, aug_combo=augments[i], lr=0.00001, n_epochs=32)

for j in range(0, 5):
    filename = f"./results/augments_experiment/combo_{1}_{j}"
    model(seed=seed, filename=filename, aug_combo=augments[0], lr=0.00001, n_epochs=32)
