# Imports
import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

DATA_LOCATION = '_data/eeg-data.csv'
NUM_EPOCHS = 150
BATCH_SIZE = 20
NUM_PREDICTIONS = 5

# Load the dataset
df = pd.read_csv(DATA_LOCATION)

# Only use subject 1 for right now
# Comment out or remove the next line to run for all subjects
df = df[df.id == 1]

# Convert from strings to array
df['eeg_power'] = df.eeg_power.map(json.loads)

# Extract eeg_power values into array of arrays
eeg_power = np.stack(df.pop('eeg_power').to_numpy())

# Extract labels into array of onehots
labels_by_onehot = {}
raw_labels = df.pop('label')
raw_onehots = pd.get_dummies(raw_labels)
for index in range(len(raw_onehots)):
    # Build dicts to convert to friendly labels
    labels_by_onehot[str(raw_onehots.values[index].tolist())] \
        = raw_labels.values[index]

labels = raw_onehots.to_numpy()


# Build model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(65, activation='sigmoid'))
model.add(Dense(len(labels_by_onehot), activation='softmax'))

# Compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the keras model on the dataset
model.fit(eeg_power, labels,
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the keras model
loss, accuracy = model.evaluate(eeg_power, labels)
print('Accuracy: {0:.2f}\nLoss: {1:.2f}'.format(accuracy*100, loss*100))

# Make probability predictions with the model
predictions = model.predict(eeg_power)

for i in range(NUM_PREDICTIONS):
    prediction = str([round(x) for x in predictions[i]])
    prediction = str(labels_by_onehot.get(prediction, "[invalid]"))
    print('{0} => {1} (expected {2})'
          .format(eeg_power[i].tolist(),
                  prediction,
                  raw_labels.values[i]))
