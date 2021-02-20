# This script uses Keras with TensorFlow to classify EEG data

## Instructions

### Setup the virtual environment

Create virtual environment: `python -m venv venv`  
Activate the virtual environment: `source venv/bin/activate` for linux/macos or `venv\Scripts\activate` for windows  
Install requirements: `pip install -r requirements.txt`

### Get the data

1. Download the [eeg dataset](https://www.kaggle.com/berkeley-biosense/synchronized-brainwave-dataset)
    > You will have to create a Kaggle account, but it is free and signup is quick.

1. Unarchive `eeg-data.csv.zip`  
1. Create a new directory in this project folder called `_data`  
1. Place the extracted `eeg-data.csv` file in the `_data` directory  
    > If you choose to use a different location, update the `DATA_LOCATION` value in the `classify-eeg-for-one.py` script

### Run the script

Run the python script: `python classify-eeg-for-one.py`

### Customize parameters

You can change the following values to adjust model parameters:

* `NUM_EPOCHS` - To adjust the number of epocs used in training
* `BATCH_SIZE` - To adjust batch size used in training
* `NUM_PREDICTIONS` - To change the number of predictions to print, this is informational only

To change model layers and layer types, see the `# Build model` section of the `classify-eeg-for-one.py` script.

You can also change the script to classify for all subjects by commenting out the `df = df[df.id == 1]` line in the `classify-eeg-for-one.py` script.
