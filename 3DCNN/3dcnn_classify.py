
import pandas
import tensorflow as tf
import numpy as np
from os import listdir
from re import split as re_split
from random import shuffle
from tensorflow import keras
from tensorflow.keras import layers

DATA_LOCATION = '_data'  # Folder containing data files
FILENAME_REGEX = '_|\\.'  # Splits files by underscore and period

# Number of columns present in the CSV files that actually contain sensor data
SENSOR_DATA_COLUMNS = 24
# Must be a factor of SENSOR_DATA_COLUMNS
TIMESLICE_ROWS = 6
# Must be the factor of SENSOR_DATA_COLUMNS complimenting TIMESLICE_ROWS
TIMESLICE_COLUMNS = 4
# Number of frames in a timeslice. Our data generates a timeslice every 0.0033s
FRAMES_PER_TIMESLICE = 20
# Percent of data that should be used for training, with the remaining
#   percentage used for validation
DATA_SPLIT_PERCENTAGE = 70

# Helpful for converting from machine readable to human readable and back
label_to_onehot = {}
onehot_to_label = {}


def get_filename_label_dict(filenames):
    filename_label_dict = {}

    for filename in filenames:
        """
        Filenames will be in the format subjectN_L.csv
        where N represents subject number and L represents
        the action the subject was taking at the time (label)

        filename_parts contains the filename split on
        underscores and periods
        """
        filename_parts = re_split(FILENAME_REGEX, filename)
        label = filename_parts[1]
        filename_label_dict[filename] = label

    return filename_label_dict


def get_input_data():
    data = pandas.DataFrame()
    filenames = listdir(DATA_LOCATION)
    filename_label_dict = get_filename_label_dict(filenames)
    distinct_labels = set(filename_label_dict.values())
    label_to_onehot, onehot_to_label = get_onehots(distinct_labels)

    for filename in filenames:
        filedata = pandas.read_csv(DATA_LOCATION + '/' + filename)
        file_label = filename_label_dict[filename]

        filedata.drop(columns=['Trigger',
                               'Time_Offset',
                               'ADC_Status',
                               'ADC_Sequence',
                               'Event',
                               'Comments'],
                      inplace=True)
        filedata['label'] = file_label

        # Assigning tuple to column value
        filedata['onehot_label'] = ([label_to_onehot[file_label]] *
                                    len(filedata))
        data = data.append(filedata)

    return data


def get_onehots(values):
    # Load this into globals to ease conversion betweek
    #   machine-readable and human-readable
    global label_to_onehot, onehot_to_label
    label_to_onehot = {}
    onehot_to_label = {}
    value_list = list(values)

    for x in range(len(value_list)):
        # Initialize tuple of length of values
        onehot_list = [0] * len(value_list)
        onehot_list[x] = 1
        onehot = tuple(onehot_list)
        label_to_onehot[value_list[x]] = onehot
        onehot_to_label[onehot] = value_list[x]

    return label_to_onehot, onehot_to_label


# Change one deminsional rows into two deminsional data frames,
#   then group those frames into timeslices organized by label.
# The remainder of (number of rows for label) / frames_per_timeslice
#   is discarded with this implementation.
def build_timeslices(data, frames_per_timeslice):
    timeslice_dict = {}
    partial_sets = {}

    # Preload dictionaries
    for label in list(set(data['onehot_label'])):
        timeslice_dict[label] = []
        partial_sets[label] = []

    for row in data.values:
        # Add one to SENSOR_DATA_COLUMS to account for time column
        #   which we are skipping.
        # Cast to float32 to fit model
        frame = row[1:SENSOR_DATA_COLUMNS + 1].astype('float32') \
                .reshape((TIMESLICE_COLUMNS, TIMESLICE_ROWS))

        # The last column in the row is the onehot label
        frame_label = row[len(row) - 1]

        partial_sets[frame_label].append(frame)

        # If a full timeslice has been constructed, add it to the dict
        if len(partial_sets[frame_label]) == frames_per_timeslice:
            timeslice_dict[frame_label].append(partial_sets[frame_label])
            partial_sets[frame_label] = []

    # BEGIN Informational only
    print("Constructed {0} timeslices of {1} frames each.".format(
        len(timeslice_dict.values()), frames_per_timeslice))

    for label in partial_sets.keys():
        number_left_over = len(partial_sets[label])
        if (number_left_over > 0):
            print("{0} frames of label {1} discarded".format(
                number_left_over, onehot_to_label[label]))
    # END Informational only

    return timeslice_dict


def get_ordered_data(timeslices):
    ordered_timeslices = []
    ordered_labels = []

    for key in timeslices.keys():
        ordered_timeslices.extend(timeslices[key])
        ordered_labels.extend((key,) * len(timeslices[key]))

    return ordered_timeslices, ordered_labels


# Shuffle two lists maintaining their order
def shuffle_together(list1, list2):
    combined = list(zip(list1, list2))
    shuffle(combined)
    list1[:], list2[:] = zip(*combined)
    return list1, list2


def split_data(data, data_split_percentage):
    split_index = round(len(data) * (data_split_percentage / 100))
    return data[:split_index], data[split_index:]


def build_model(columns, rows, depth):
    # Based on: https://keras.io/examples/vision/3D_image_classification/

    inputs = keras.Input((depth, columns, rows, 1))
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=3, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3DCNN")
    return model


def define_data_loaders(train_data, train_labels, validation_data,
                        validation_labels):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((train_data,
                                                       train_labels))
    validation_loader = tf.data.Dataset.from_tensor_slices((validation_data,
                                                            validation_labels))

    batch_size = 2

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(train_data))
        # .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(validation_data))
        # .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    return train_dataset, validation_dataset


def train_model(model, train_dataset, validation_dataset):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=["acc"])

    # Define callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_attention_classification.h5", save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc",
                                                      patience=15)

    # Train the model, doing validation after each epoch
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        # shuffle = True, # Omitting since I already shuffled the data
        callbacks=[checkpoint_cb, early_stopping_cb]
    )


def make_predictions(model, validation_data, validation_labels,
                     prediction_index):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    # Load best weights
    prediction_scores = []

    model.load_weights("3d_attention_classification.h5")
    prediction = model.predict(
        np.expand_dims(validation_data[prediction_index],
                       axis=0))[0]

    for key in onehot_to_label.keys():
        index = key.index(1)
        prediction_scores.append((prediction[index], onehot_to_label[key], ))

    for score in prediction_scores:
        print(
            "This model is {0:.2f} percent confident that label is {1}"
            .format((100 * score[0]), score[1])
        )
    print("The actual label is: " +
          onehot_to_label[validation_labels[prediction_index]])


data = get_input_data()
timeslice_dict = build_timeslices(data, FRAMES_PER_TIMESLICE)
ordered_timeslices, ordered_labels = get_ordered_data(timeslice_dict)
shuffled_timeslices, shuffled_labels = shuffle_together(ordered_timeslices,
                                                        ordered_labels)
train_labels, validation_labels = split_data(shuffled_labels,
                                             DATA_SPLIT_PERCENTAGE)
train_data, validation_data = split_data(shuffled_timeslices,
                                         DATA_SPLIT_PERCENTAGE)

print("\nThe number of training samples is {0}".format(len(train_labels)))
print("The number of validation samples is {0}\n"
      .format(len(validation_labels)))

train_dataset, validation_dataset = define_data_loaders(train_data,
                                                        train_labels,
                                                        validation_data,
                                                        validation_labels)

model = build_model(TIMESLICE_COLUMNS, TIMESLICE_ROWS, FRAMES_PER_TIMESLICE)
model.summary()

train_model(model, train_dataset, validation_dataset)

make_predictions(model, validation_data, 0)
