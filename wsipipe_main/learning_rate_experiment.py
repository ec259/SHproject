from resnet import run as model

seed = 99
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

for i in range(1, 5):
        filename = f"./results/lr_experiment/" + str(learning_rates[i])
        model(seed=seed, filename=filename, aug_combo=None, lr=learning_rates[i], n_epochs=10)
