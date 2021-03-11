# Folder containing data files
DATA_LOCATION = '_data_stew_3_level'

# Number of columns present in the CSV files that actually contain sensor data
SENSOR_DATA_COLUMNS = 16

# Must be a factor of SENSOR_DATA_COLUMNS
TIMESLICE_ROWS = 4

# Must be the factor of SENSOR_DATA_COLUMNS complimenting TIMESLICE_ROWS
TIMESLICE_COLUMNS = 4

# Must evenly distribute along rows and columns
KERNEL_SIZE = 2

# Number of frames in a timeslice. Our data generates a timeslice every 0.0033s
FRAMES_PER_TIMESLICE = 20

# Percent of data that should be used for training, with the remaining
#   percentage used for validation
DATA_SPLIT_PERCENTAGE = 70

# Max number of epochs. Training will stop early if no improvement after
#   TRAINING_PATIENCE number of epochs
MAX_EPOCHS = 100

# Number of epochs with no improvement before training stops early
TRAINING_PATIENCE = 15

# Number of predictions to make after completing model training
PREDICTIONS = 10

# Trained model filename
MODEL_FILE_NAME = '3d_attention_classification_stew.h5'
