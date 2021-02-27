
import pandas
# import numpy
from os import listdir
from re import split as re_split

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
        #   which we are skipping
        frame = row[1:SENSOR_DATA_COLUMNS + 1].reshape(
            (TIMESLICE_COLUMNS, TIMESLICE_ROWS))

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
        ordered_timeslices.append(timeslices[key])
        ordered_labels.extend((key,) * len(timeslices[key]))

    return ordered_timeslices, ordered_labels


data = get_input_data()
timeslices = build_timeslices(data, FRAMES_PER_TIMESLICE)
ordered_timeslices, ordered_labels = get_ordered_data(timeslices)
