# flake8: noqa
from classify import do_classification
import settings

global passive, passive_fourier, passive_fourier_append, active, active_fourier, active_fourier_append

force_training = False

drop_cols_16 = [*settings.DROP_COLUMNS, "X3", "X2", "X1", "O1", "O2", "A1", "A2", "CM"]
drop_cols_12 = [*drop_cols_16, "Fp1", "Fp2", "T5", "T6"]
drop_cols_6 = [*drop_cols_12, "F7", "F8", "T3", "T4", "P3", "P4"]
drop_cols_3 = [*drop_cols_6, "F3", "F4", "Cz"]

# Run tests
passive = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_passive', model_file_name='passive.h5', model_weights_file_name='passive_weights.h5', use_fourier=False, fourier_append=False)
passive_fourier = do_classification(force_training=force_training, max_epochs=100, early_stopping=False, batch_size=32, frames_per_timeslice=300, data_location='_data_passive', model_file_name='passive_fourier.h5', model_weights_file_name='passive_fourier_weights.h5', use_fourier=True, fourier_append=False)
passive_fourier_append = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=300, data_location='_data_passive', model_file_name='passive_fourier_append.h5', model_weights_file_name='passive_fourier_append_weights.h5', use_fourier=True, fourier_append=True)
passive_16_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_passive', model_file_name='passive_16_col.h5', model_weights_file_name='passive_16_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_16, num_columns=4, num_rows=4)
passive_12_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_passive', model_file_name='passive_12_col.h5', model_weights_file_name='passive_12_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_12, num_columns=4, num_rows=3)
passive_6_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_passive', model_file_name='passive_6_col.h5', model_weights_file_name='passive_6_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_6, num_columns=3, num_rows=2)
passive_3_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_passive', model_file_name='passive_3_col.h5', model_weights_file_name='passive_3_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_3, num_columns=3, num_rows=1)
active = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_active', model_file_name='active.h5', model_weights_file_name='active_weights.h5', use_fourier=False, fourier_append=False)
active_fourier = do_classification(force_training=force_training, max_epochs=100, early_stopping=False, batch_size=32, frames_per_timeslice=300, data_location='_data_active', model_file_name='active_fourier.h5', model_weights_file_name='active_fourier_weights.h5', use_fourier=True, fourier_append=False)
active_fourier_append = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=300, data_location='_data_active', model_file_name='active_fourier_append.h5', model_weights_file_name='active_fourier_append_weights.h5', use_fourier=True, fourier_append=True)
active_16_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_active', model_file_name='active_16_col.h5', model_weights_file_name='active_16_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_16, num_columns=4, num_rows=4)
active_12_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_active', model_file_name='active_12_col.h5', model_weights_file_name='active_12_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_12, num_columns=4, num_rows=3)
active_6_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_active', model_file_name='active_6_col.h5', model_weights_file_name='active_6_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_6, num_columns=3, num_rows=2)
active_3_col = do_classification(force_training=force_training, max_epochs=20, early_stopping=False, batch_size=32, frames_per_timeslice=20, data_location='_data_active', model_file_name='active_3_col.h5', model_weights_file_name='active_3_col_weights.h5', use_fourier=False, fourier_append=False, drop_columns=drop_cols_3, num_columns=3, num_rows=1)

models = {
    "Passive": passive,
    "Passive Fourier": passive_fourier,
    "Passive Fourier Append": passive_fourier_append,
    "Passive 16 Col": passive_16_col,
    "Passive 12 Col": passive_12_col,
    "Passive 6 Col": passive_6_col,
    "Passive 3 Col": passive_3_col,
    "Active": active,
    "Active Fourier": active_fourier,
    "Active Fourier Append": active_fourier_append,
    "Active 16 Col": active_16_col,
    "Active 12 Col": active_12_col,
    "Active 6 Col": active_6_col,
    "Active 3 Col": active_3_col,
}

# Calculate accuracy and error
max_len = max([len(key) for key in models.keys()])
results = {}
for key in models.keys():
    train, validation = models[key].get_accuracy(verbose=False)
    prediction_error = models[key].get_average_prediction_error()

    results[key] = ("| {key:{max_len}} | {train:6.2f}% | {val:6.2f}% | {pred_err:6.2f}% |"
        .format(key=key, max_len=max_len, train=train*100, val=validation*100, pred_err=prediction_error*100))

# Print table
print("\n\n\nResults:")
header = "| {0:{max_len}} | {1:>7} | {2:>7} | {3:>7} |".format('MODEL', 'TRN ACC', 'VAL ACC', 'AVG ERR', max_len=max_len)
print("-"*len(header))
print(header)
[print(result) for result in results.values()]
print("-"*len(header))