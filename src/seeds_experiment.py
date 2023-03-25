from resnet import run as model


seeds = [0, 42, 123, 301, 256, 999, 647, 1011, 598, 64]
for seed in seeds:
    for i in range(0, 10):
        filename = f"./results/seeds/seed_{seed}_{i}"
        model(seed=seed, filename=filename, aug_combo=None, lr=0.00001, n_epochs=32)
