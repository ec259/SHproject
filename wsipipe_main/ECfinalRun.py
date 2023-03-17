from ECresnet import run as model

for i in range(0, 5):
    filename = f"./results/TEST_RESULTS_" + str(i)
    model(seed=99, filename=filename, aug_combo=None, lr=0.00001, n_epochs=32)
