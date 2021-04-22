# 3DCNN

## Installation

This program has only been tested using **Python 3.8**. Other versions will require you to manually adjust the requirements listed in requirements.txt. I have tried using both Python 3.7 and Python 3.9 but ultimately gave up. It is much easier to download a Python 3.8 binary from [here](https://www.python.org/downloads/) and use it to create a virtual environment as suggested below.

1. Open a command prompt or terminal window
1. Clone this project to the directory of your choice
1. (Optional, but highly recommended:) Create a virtual environment by executing `python -m venv venv`
1. Upgrade pip by executing `pip install --upgrade pip`
1. Install all requirements by executing `pip install -r requirements.txt`
1. Activate the virtual environment:
   * If using MacOS/Linux, execute `source ./venv/bin/activate`
   * If using Windows, execute `.\venv\Scripts\activate`

## Setup

Setup differs based on what data you wish to classify.

### Setup for DSI-24 data

1. Download the EEG data captured using the DSI-24
1. If the data is compressed in a .zip file, extract it
1. Create a directory called `_data` in the cloned project folder where the EEG data files will be placed
1. Move the data files that you wish to classify into the `_data` directory
1. Ensure that the filenames match the format `subjectN_L.csv` where N is the subject number and L is the label for the data
1. Execute the following in a command prompt or terminal window to remove unneeded header information from the files: `python preprocess_files.py`

### Setup for STEW data

1. Download the STEW data from [here](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset#files)

> You will need to create a free account to download the files

1. Extract the compressed .zip file
1. Create a directory called `_data_stew` in the cloned project folder
1. Move all files in the extracted STEW data folder except for `ratings.txt` to the `_data_stew` directory

> If you want to classify based on three levels (low/medium/high) instead of the default low/high, you will need to update the file names to change "lo" or "hi" to "me" based on the ratings given in the ratings.txt file

## Running

1. Navigate to the cloned project folder in a command prompt or terminal window if you have not already done so
1. Activate the virtual environment if you have not already done so:
   * If using MacOS/Linux, execute `source venv/bin/activate`
   * If using Windows, execute `venv\Scripts\activate`
1. Execute `python` to initialize the interpreter
1. Execute `import classify` if you wish to classify DSI-24 data, or `import classify_stew as classify` if you want to classify STEW data
1. Execute `context = classify.do_classification()` to classify data with default parameters and save capture the resulting context
   * Use the parameter `force_training` to "True" force training rather than use any previously existing model weights or "False" (default) to load existing model weights when they exist.
   * Use the parameter `max_epochs` to override the MAX_EPOCHS value found in the settings file.
   * Use the parameter `batch_size` to Override the BATCH_SIZE value found in the settings file.
   * Use the parameter `frames_per_timeslice` to override the FRAMES_PER_TIMESLICE value found in the settings file.
   * Use the parameter `data_location` to override the DATA_LOCATION value found in the settings file.
   * Use the parameter `model_file_name` to override the MODEL_FILE_NAME value found in the settings file.
   * Use the parameter `model_weights_file_name` to override the MODEL_WEIGHTS_FILE_NAME value found in the settings file.
   * Use the parameter `transform` to "fourier" or "wvd" for Fourier or WVD transformation, or None (default) for no transformation. This may change number of columns and rows automatically.
   * Use the parameter `transform_append` to set whether fourier transformation data should be appended to raw data or replace it (default).
   * Use the parameter `fourier_eeg_bands` to override the fourier.DEFAULT_EEG_BANDS value found in the settings file.
   * Use the parameter `early_stopping` to "True" to enable or "False" (default) to disable early stopping during training.
   * Use the parameter `num_columns` to override the TIMESLICE_COLUMNS value found in the settings file.
   * Use the parameter `num_rows` to override the TIMESLICE_ROWS value found in the settings file.
   * Use the parameter `drop_columns` to override the DROP_COLUMNS value found in the settings file.
   * Use the parameter `kernel_size` to override the KERNEL_SIZE value found in the settings file.
1. Once the training (if applicable) is complete, use the following commands to make predictions:
   * Execute `context.make_predictions(N)` to make N predictions, the program will output the level of certainty for each label. You can include the parameter `offset=X` to skip X number of data points before making the predictions
   * Execute `context.check_predictions(N)` to check N predictions, the program will only output when the predictions are incorrect. You could then use `make_predictions` (above) with an offset to determine the program's certainty of each label if desired. You can include the parameter `offset=X` to skip X number of data points before making the predictions. You can include the parameter `verbose=True` to force the program to output when predictions are correct as well as incorrect.
   * Execute `context.get_accuracy()` to get the model's accuracy.
   * Execute `context.get_average_prediction_error()` to calculate and return the average error over all validation data predictions.

## Advanced Usage

To use different data or change how the program interacts with given data, check out the `settings.py` (and `settings_stew.py`) file. You can't change everything there, for example to adjust the model you would still have to update the script itself, but you can make changes to run the same model on differently structured data.

NOTE: You must install matplotlib using `pip install matplotlib` to generate graphs. It is currently not included in the requirements.txt file.
