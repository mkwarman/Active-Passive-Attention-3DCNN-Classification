# first neural network with keras tutorial:
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# 0: Imports
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

NUM_EPOCHS = 150
BATCH_SIZE = 10
NUM_PREDICTIONS = 5

# 1: Load the dataset
dataset = loadtxt('_data/pima-indians-diabetes.csv', delimiter=',')
# split into input and output variables
dataset_input = dataset[:, 0:8]
dataset_output = dataset[:, 8]

# 2: Define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3: Compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4: Fit the keras model on the dataset
model.fit(dataset_input, dataset_output,
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# 5: Evaluate the keras model
loss, accuracy = model.evaluate(dataset_input, dataset_output)
print('Accuracy: {0:.2f}\nLoss: {1:.2f}'.format(accuracy*100, loss*100))

# 6: (Run the program up to here)

# 7: Make probability predictions with the model
predictions = model.predict(dataset_input)
# Round predictions
rounded = [round(prediction[0]) for prediction in predictions]
# Alternatively can do: predictions = model.predict_classes(dataset_input)

for i in range(NUM_PREDICTIONS):
    print('{0} => {1} (expected {2})'.format(dataset_input[i].tolist(),
                                             rounded[i],
                                             dataset_output[i]))
