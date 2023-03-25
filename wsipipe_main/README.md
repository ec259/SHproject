# Deep Learning for Camelyon 17
### Evie Currie
Files cannot be run using wsipipe dependency until pip is updated with Camelyon 17 infrastructure. Files provided in the wsipipe directory are new additions waiting to be integrated into pip.

### Deep Learning Model
 
- data_processing.py - Prepares the data for deep learning
- augmentation_transforms.py - Set of image augmentations in the transform format ready to be utilised by the model.
- resnet.py - The deep learning model (ResNet50 architecture)
- start.sh - Starts venv and installs all requirements. Runs data pre-processing (creat ing patches)

### Investigations
- augmentations_experiment.py - Runs experiment to compare accuracy results of all augmentation combinations.
- learning_rate_experiment.py - Runs experiment to compare accuracy results of 5 different learning rates on the model.
- seeds_experiment.py - Runs experiment to compare accuracy results of different seeds (as well as see variance within seeds)

### Results
The Results folder contains the results from all of the listed experiments above. Python scripts for plots/graphsa re also included as well as outputted png's from the graphing scripts.