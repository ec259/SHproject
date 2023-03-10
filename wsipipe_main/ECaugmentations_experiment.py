from ECresnet import run as model
import ECaugments

seed = 99
for seed in seeds:
    for i in range(0, 10):
        filename = f"./results/seeds/seed_{seed}_{i}"
        model(seed=seed, filename=filename, aug_combo=None, lr=0.00001, n_epochs=32)
