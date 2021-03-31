# flake8: noqa
from multiprocessing import Pool
from classify import do_classification
import settings

PROCESSES = 8
global passive, passive_fourier, passive_fourier_append, active, active_fourier, active_fourier_append

force_training = False

drop_cols_16 = [*settings.DROP_COLUMNS, "X3", "X2", "X1", "O1", "O2", "A1", "A2", "CM"]
drop_cols_12 = [*drop_cols_16, "Fp1", "Fp2", "T5", "T6"]
drop_cols_6 = [*drop_cols_12, "F7", "F8", "T3", "T4", "P3", "P4"]
drop_cols_3 = [*drop_cols_6, "F3", "F4", "Cz"]
expanded_eeg_bands = {'Delta_1': (0, 1),
                      'Delta_2': (1, 2),
                      'Delta_3': (2, 3),
                      'Delta_4': (3, 4),
                      'Theta_1': (4, 5),
                      'Theta_2': (5, 6),
                      'Theta_3': (6, 7),
                      'Theta_4': (7, 8),
                      'Alpha_1': (8, 9),
                      'Alpha_2': (9, 10),
                      'Alpha_3': (10, 11),
                      'Alpha_4': (11, 12),
                      'Beta_1': (12, 16.5),
                      'Beta_2': (16.5, 21),
                      'Beta_3': (21, 25.5),
                      'Beta_4': (25.5, 30),
                      'Gamma_1': (30, 33.75),
                      'Gamma_2': (33.75, 37.5),
                      'Gamma_3': (37.5, 41.25),
                      'Gamma_4': (41.25, 45)}

# Setup Args
model_args = [
    ("passive", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_passive', "model_file_name":'passive.h5', "model_weights_file_name":'passive_weights.h5', "use_fourier":False, "fourier_append":False}),
    ("passive_fourier", {"force_training":force_training, "max_epochs":100, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_passive', "model_file_name":'passive_fourier.h5', "model_weights_file_name":'passive_fourier_weights.h5', "use_fourier":True, "fourier_append":False}),
    ("passive_fourier_append", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_passive', "model_file_name":'passive_fourier_append.h5', "model_weights_file_name":'passive_fourier_append_weights.h5', "use_fourier":True, "fourier_append":True}),
    ("passive_fourier_expanded", {"force_training":force_training, "max_epochs":100, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_passive', "model_file_name":'passive_fourier_expanded.h5', "model_weights_file_name":'passive_fourier_expanded_weights.h5', "use_fourier":True, "fourier_append":False, "fourier_eeg_bands":expanded_eeg_bands}),
    ("passive_fourier_expanded_append", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_passive', "model_file_name":'passive_fourier_expanded_append.h5', "model_weights_file_name":'passive_fourier_expanded_append_weights.h5', "use_fourier":True, "fourier_append":True, "fourier_eeg_bands":expanded_eeg_bands}),
    ("passive_16_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_passive', "model_file_name":'passive_16_col.h5', "model_weights_file_name":'passive_16_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_16, "num_columns":4, "num_rows":4}),
    ("passive_12_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_passive', "model_file_name":'passive_12_col.h5', "model_weights_file_name":'passive_12_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_12, "num_columns":4, "num_rows":3}),
    ("passive_6_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_passive', "model_file_name":'passive_6_col.h5', "model_weights_file_name":'passive_6_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_6, "num_columns":3, "num_rows":2}),
    ("passive_3_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_passive', "model_file_name":'passive_3_col.h5', "model_weights_file_name":'passive_3_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_3, "num_columns":3, "num_rows":1}),
    ("active", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_active', "model_file_name":'active.h5', "model_weights_file_name":'active_weights.h5', "use_fourier":False, "fourier_append":False}),
    ("active_fourier", {"force_training":force_training, "max_epochs":100, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_active', "model_file_name":'active_fourier.h5', "model_weights_file_name":'active_fourier_weights.h5', "use_fourier":True, "fourier_append":False}),
    ("active_fourier_append", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_active', "model_file_name":'active_fourier_append.h5', "model_weights_file_name":'active_fourier_append_weights.h5', "use_fourier":True, "fourier_append":True}),
    ("active_fourier_expanded", {"force_training":force_training, "max_epochs":100, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_active', "model_file_name":'active_fourier_expanded.h5', "model_weights_file_name":'active_fourier_expanded_weights.h5', "use_fourier":True, "fourier_append":False, "fourier_eeg_bands":expanded_eeg_bands}),
    ("active_fourier_expanded_append", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":300, "data_location":'_data_active', "model_file_name":'active_fourier_expanded_append.h5', "model_weights_file_name":'active_fourier_expanded_append_weights.h5', "use_fourier":True, "fourier_append":True, "fourier_eeg_bands":expanded_eeg_bands}),
    ("active_16_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_active', "model_file_name":'active_16_col.h5', "model_weights_file_name":'active_16_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_16, "num_columns":4, "num_rows":4}),
    ("active_12_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_active', "model_file_name":'active_12_col.h5', "model_weights_file_name":'active_12_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_12, "num_columns":4, "num_rows":3}),
    ("active_6_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_active', "model_file_name":'active_6_col.h5', "model_weights_file_name":'active_6_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_6, "num_columns":3, "num_rows":2}),
    ("active_3_col", {"force_training":force_training, "max_epochs":20, "early_stopping":False, "batch_size":32, "frames_per_timeslice":20, "data_location":'_data_active', "model_file_name":'active_3_col.h5', "model_weights_file_name":'active_3_col_weights.h5', "use_fourier":False, "fourier_append":False, "drop_columns":drop_cols_3, "num_columns":3, "num_rows":1})
]

max_len = max([len(model_arg[0]) for model_arg in model_args])
def compute(model_arg):
    result = do_classification(**model_arg[1])
    train, validation = result.get_accuracy(verbose=False)
    prediction_error = result.get_average_prediction_error()

    return ("| {key:{max_len}} | {train:6.2f}% | {val:6.2f}% | {pred_err:6.2f}% |"
            .format(key=model_arg[0], max_len=max_len, train=train*100, val=validation*100, pred_err=prediction_error*100))

if __name__ == '__main__':
    results = []
    with Pool(PROCESSES) as pool:
        results = pool.map(compute, model_args)

    print("\n\n\nResults:")
    header = "| {0:{max_len}} | {1:>7} | {2:>7} | {3:>7} |".format('MODEL', 'TRN ACC', 'VAL ACC', 'AVG ERR', max_len=max_len)
    print("-"*len(header))
    print(header)
    [print(result) for result in results]
    print("-"*len(header))
