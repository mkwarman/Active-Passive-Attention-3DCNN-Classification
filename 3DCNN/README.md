# 3DCNN

## Installation:
**You must use Python 3.8**

1. Open a command prompt or terminal window
1. Clone this project to the directory of your choice
1. (Optional, but highly recommended:) Create a virtual environment by executing `python -m venv venv`
1. Upgrade pip by executing `pip install --upgrade pip`
1. Install all requirements by executing `pip install -r requirements.txt`
1. Activate the virtual environment:
   * If using MacOS/Linux, execute `source ./venv/bin/activate`
   * If using Windows, execute `.\venv\Scripts\activate`

## Setup:
Setup differs based on what data you wish to classify.

### Setup for DSI-24 data:
1. Download the EEG data captured using the DSI-24
1. If the data is compressed in a .zip file, extract it
1. Create a directory called `_data` in the cloned project folder where the EEG data files will be placed
1. Move the data files that you wish to classify into the `_data` directory
1. Ensure that the filenames match the format `subjectN_L.csv` where N is the subject number and L is the label for the data
1. Execute the following in a command prompt or terminal window to remove unneeded header information from the files: `python preprocess_files.py`

### Setup for STEW data:
1. Download the STEW data from [here](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset#files)
> You will need to create a free account to download the files
1. Extract the compressed .zip file
1. Create a directory called `_data_stew` in the cloned project folder
1. Move all files in the extracted STEW data folder except for `ratings.txt` to the `_data_stew` directory
> If you want to classify based on three levels (low/medium/high) instead of the default low/high, you will need to update the file names to change "lo" or "hi" to "me" based on the ratings given in the ratings.txt file

## Running:
1. Navigate to the cloned project folder in a command prompt or terminal window if you have not already done so
1. Activate the virtual environment if you have not already done so:
   * If using MacOS/Linux, execute `source venv/bin/activate`
   * If using Windows, execute `venv\Scripts\activate`
1. Execute `python` to initialize the interpreter
1. Execute `import classify` if you wish to classify DSI-24 data, or `import classify_stew as classify` if you want to classify STEW data
1. Execute `context = classify.do_classification()` to classify data with default parameters and save capture the resulting context
   * Use the parameter `force_training=True` to force training rather than use any previously existing model weights
   * Use the parameter `max_epochs=N` to override the MAX_EPOCHS value found in the settings file
   * Use the parameter `batch_size=N` to Override the BATCH_SIZE value found in the settings file
1. Once the training (if applicable) is complete, use the following commands to make predictions:
   * Execute `context.make_predictions(N)` to make N predictions, the program will output the level of certainty for each label. You can include the parameter `offset=X` to skip X number of data points before making the predictions
   * Execute `context.check_predictions(N)` to check N predictions, the program will only output when the predictions are incorrect. You could then use `make_predictions` (above) with an offset to determine the program's certainty of each label if desired. You can include the parameter `offset=X` to skip X number of data points before making the predictions. You can include the parameter `verbose=True` to force the program to output when predictions are correct as well as incorrect.

## Advanced Usage:
To use different data or change how the program interacts with given data, check out the `settings.py` (and `settings_stew.py`) file. You can't change everything there, for example to adjust the model you would still have to update the script itself, but you can make changes to run the same model on differently structured data.