
import pandas
# import numpy
from os import listdir
from re import split as re_split

DATA_LOCATION = '_data'  # Folder containing data files
FILENAME_REGEX = '_|\\.'  # Splits files by underscore and period


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
        filedata[len(filedata.columns)] = file_label
        filedata[len(filedata.columns)] = str(label_to_onehot[file_label])
        data = data.append(filedata)

    return data


def get_onehots(values):
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


data = get_input_data()
