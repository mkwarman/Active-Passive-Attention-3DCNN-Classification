import pandas
import tensorflow as tf
import numpy as np
import fourier
import settings
from os import listdir, path
from re import split as re_split
from random import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model
from tqdm import tqdm
from classification_context import ClassificationContext

FILENAME_REGEX = '-|\\.'  # Splits files by dash and period


def get_filename_label_dict(filenames):
    filename_label_dict = {}

    for filename in filenames:
        """
        Filenames will be in the format subjectN-L.csv
        where N represents subject number and L represents
        the action the subject was taking at the time (label)

        filename_parts contains the filename split on
        underscores and periods
        """
        filename_parts = re_split(FILENAME_REGEX, filename)
        label = filename_parts[1]
        filename_label_dict[filename] = label

    return filename_label_dict


def get_onehots(values):
    label_to_onehot = {}
    onehot_to_label = {}

    # Sort to ensure that when loading a previous model we keep the labels
    #   in the correct order
    value_list = sorted(values)

    for x in range(len(value_list)):
        # Initialize tuple of length of values
        onehot_list = [0] * len(value_list)
        onehot_list[x] = 1
        onehot = tuple(onehot_list)
        label_to_onehot[value_list[x]] = onehot
        onehot_to_label[onehot] = value_list[x]

    return label_to_onehot, onehot_to_label


def get_input_data(data_location, drop_columns):
    data = []
    filenames = listdir(data_location)

    # Get rid of things like .DS_Store
    filenames = list(filter(lambda f: not f.startswith("."), filenames))
    filename_label_dict = get_filename_label_dict(filenames)
    distinct_labels = set(filename_label_dict.values())
    label_to_onehot, onehot_to_label = get_onehots(distinct_labels)

    print("Loading input data files:")
    for filename in tqdm(filenames):
        filedata = pandas.read_csv(data_location + '/' + filename)

        # Remove whitespace from column names
        filedata = filedata.rename(columns=lambda col: col.strip())

        file_label = filename_label_dict[filename]

        filedata.drop(columns=drop_columns, inplace=True)
        filedata['label'] = file_label

        # Assigning tuple to column value
        filedata['onehot_label'] = ([label_to_onehot[file_label]] *
                                    len(filedata))
        data.append(filedata)

    print("Loaded all data successfully.\n")

    frame = pandas.concat(data, axis=0, ignore_index=True)
    return frame, label_to_onehot, onehot_to_label


# Change one deminsional rows into two deminsional data frames,
#   then group those frames into timeslices organized by label.
# The remainder of (number of rows for label) / frames_per_timeslice
#   is discarded with this implementation.
def build_timeslices(data, frames_per_timeslice, onehot_to_label,
                     num_columns, num_rows):
    timeslice_dict = {}
    partial_sets = {}

    # Preload dictionaries
    for label in list(set(data['onehot_label'])):
        timeslice_dict[label] = []
        partial_sets[label] = []

    for row in data.values:
        """
        temp = row[:len(row) - 2]
        target_len = num_columns * num_rows
        if (len(temp) < target_len):
            zeros_to_add = target_len - len(temp)
            zeros = list((0,) * zeros_to_add)
            temp = np.concatenate((temp, zeros))

        frame = (temp.astype('float32')
                 .reshape((num_columns, num_rows)))
        """
        # Cast to float32 to fit model and reshape, excluding label columns
        frame = (row[:len(row) - 2].astype('float32')
                 .reshape((num_columns, num_rows)))

        # The last column in the row is the onehot label
        frame_label = row[len(row) - 1]

        partial_sets[frame_label].append(frame)

        # If a full timeslice has been constructed, stack in sequence
        #   and add it to the dict
        if len(partial_sets[frame_label]) == frames_per_timeslice:
            timeslice_dict[frame_label].append(
               np.stack(partial_sets[frame_label]))
            partial_sets[frame_label] = []

    # BEGIN Informational only
    for key in timeslice_dict.keys():
        print("Constructed {0} timeslices of label \"{1}\" with {2} "
              "frames each.".format(len(timeslice_dict[key]),
                                    onehot_to_label[key],
                                    frames_per_timeslice))

    for label in partial_sets.keys():
        number_left_over = len(partial_sets[label])
        if (number_left_over > 0):
            print("{0} frames of label {1} discarded".format(
                number_left_over, onehot_to_label[label]))
    # END Informational only

    return timeslice_dict


def get_fourier_data_for_frames(frames, eeg_bands, num_columns, num_rows):
    # Create stack of planes for each eeg band with values corresponding to
    #   each sensor
    fft_bands = np.zeros((len(eeg_bands), num_columns, num_rows))
    for i in range(num_columns):
        for j in range(num_rows):
            band_data = list(fourier.partition_eeg_bands(
                frames[:, i, j], settings.SENSOR_HERTZ,
                eeg_bands=eeg_bands).values())
            for band in range(len(fft_bands)):
                fft_bands[band, i, j] = band_data[band]

    return fft_bands


def load_fourier_data(timeslice_dict, num_columns, num_rows, eeg_bands,
                      append=False):
    keys = list(timeslice_dict.keys())

    fft_data = {}
    for key in keys:
        fft_data[key] = []
        for frames in timeslice_dict[key]:
            fft_frames = get_fourier_data_for_frames(frames, eeg_bands,
                                                     num_columns, num_rows)

            if append:
                frames_with_fft_frames = np.append(frames, fft_frames, 0)
                fft_data[key].append(frames_with_fft_frames)
            else:
                fft_data[key].append(fft_frames)

    return fft_data


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


def build_model(columns, rows, depth, output_units):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    pool_size = 2
    kernel_size = settings.KERNEL_SIZE

    # Handle small dimensions
    if (columns <= 2 or rows <= 2):
        pool_size = 1
        kernel_size = 1

    # Handle odd dimension lengths
    if ((columns % 2 != 0) or (rows % 2 != 0)):
        kernel_size = 1

    inputs = keras.Input((depth, columns, rows, 1))
    x = layers.Conv3D(filters=64,
                      kernel_size=kernel_size,
                      activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=pool_size)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=output_units, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3DCNN")
    return model


def define_data_loaders(train_data, train_labels, validation_data,
                        validation_labels, batch_size):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((train_data,
                                                       train_labels))
    validation_loader = tf.data.Dataset.from_tensor_slices((validation_data,
                                                            validation_labels))

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


def train_model(model, train_dataset, validation_dataset, max_epochs,
                model_file_name, model_weights_file_name, early_stopping):
    # Based on: https://keras.io/examples/vision/3D_image_classification/
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=["acc"])
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=[tf.keras.metrics.CategoricalCrossentropy()])
    save_model(model, model_file_name)

    # Define callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_weights_file_name, save_best_only=True)

    callbacks = [checkpoint_cb]

    if early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_acc",
            # monitor="val_categorical_crossentropy",
            patience=settings.TRAINING_PATIENCE)
        callbacks.append(early_stopping_cb)

    # Train the model, doing validation after each epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=max_epochs,
        # shuffle = True,  # Omitting since I already shuffled the data
        callbacks=callbacks
    )

    # Load best weights
    model.load_weights(model_weights_file_name)


def do_classification(force_training=False,
                      max_epochs=settings.MAX_EPOCHS,
                      batch_size=settings.BATCH_SIZE,
                      frames_per_timeslice=settings.FRAMES_PER_TIMESLICE,
                      data_location=settings.DATA_LOCATION,
                      model_file_name=settings.MODEL_FILE_NAME,
                      model_weights_file_name=settings.MODEL_WEIGHTS_FILE_NAME,
                      use_fourier=False,
                      fourier_append=False,
                      fourier_eeg_bands=fourier.DEFAULT_EEG_BANDS,
                      early_stopping=True,
                      num_columns=settings.TIMESLICE_COLUMNS,
                      num_rows=settings.TIMESLICE_ROWS,
                      drop_columns=settings.DROP_COLUMNS):
    """
    # Force square
    if num_columns != num_rows:
        num = max([num_columns, num_rows])
        num_columns = num
        num_rows = num
    """

    context = ClassificationContext()

    data, label_to_onehot, onehot_to_label = get_input_data(data_location,
                                                            drop_columns)
    context.set_label_onehot(label_to_onehot, onehot_to_label)

    print("Building timeslices...")

    timeslice_dict = build_timeslices(data, frames_per_timeslice,
                                      onehot_to_label, num_columns,
                                      num_rows)

    if use_fourier:
        print("\nComputing Fourier tranforms...")
        timeslice_dict = load_fourier_data(timeslice_dict, num_columns,
                                           num_rows, fourier_eeg_bands,
                                           append=fourier_append)
        frames_per_timeslice = (len(fourier_eeg_bands) + frames_per_timeslice
                                if fourier_append else len(fourier_eeg_bands))

    context.set_timeslice_dict(timeslice_dict)

    ordered_timeslices, ordered_labels = get_ordered_data(timeslice_dict)
    shuffled_timeslices, shuffled_labels = shuffle_together(ordered_timeslices,
                                                            ordered_labels)
    train_labels, validation_labels = \
        split_data(shuffled_labels, settings.DATA_SPLIT_PERCENTAGE)
    train_data, validation_data = \
        split_data(shuffled_timeslices, settings.DATA_SPLIT_PERCENTAGE)
    context.set_data(train_data, train_labels,
                     validation_data, validation_labels)

    print("\nThe number of training samples is {0}"
          .format(len(train_labels)))
    print("The number of validation samples is {0}\n"
          .format(len(validation_labels)))

    print("Building datasets...\n")

    train_dataset, validation_dataset = define_data_loaders(train_data,
                                                            train_labels,
                                                            validation_data,
                                                            validation_labels,
                                                            batch_size)
    context.set_datasets(train_dataset, validation_dataset)

    model = None
    if force_training or not (path.exists(model_file_name) and
                              path.exists(model_weights_file_name)):
        print("Building model...\n")
        model = build_model(num_columns,
                            num_rows,
                            frames_per_timeslice,
                            len(onehot_to_label.keys()))
        model.summary()
        train_model(model, train_dataset, validation_dataset, max_epochs,
                    model_file_name, model_weights_file_name, early_stopping)
        print("\nTraining complete.\n")
    else:
        model = load_model(model_file_name)
        print("\nLoaded model from {0}\n"
              .format(model_file_name))
        model.summary()
        model.load_weights(model_weights_file_name)
        print("\nLoaded weights from {0}\n"
              .format(model_weights_file_name))

    context.set_model(model)

    return context


if (__name__ == '__main__'):
    do_classification()
