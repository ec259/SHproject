from ECresnet import run as model

for i in range(0, 1):
    filename = f"./results/TEST_RESULTS_DIFF_LR" + str(i)
    model(seed=999, filename=filename, aug_combo=None, lr=0.0001, n_epochs=32)
