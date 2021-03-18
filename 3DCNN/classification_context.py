import numpy as np
from tqdm import trange


class ClassificationContext:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.validation_data = []
        self.validation_labels = []
        self.train_dataset = []
        self.validation_dataset = []
        self.label_to_onehot = {}
        self.onehot_to_label = {}
        self.model = None

    def set_data(self, train_data, train_labels,
                 validation_data, validation_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def set_datasets(self, train_dataset, validation_dataset):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def set_model(self, model):
        self.model = model

    def set_label_onehot(self, label_to_onehot, onehot_to_label):
        self.label_to_onehot = label_to_onehot
        self.onehot_to_label = onehot_to_label

    def predict_for_index(self, prediction_index):
        # Based on: https://keras.io/examples/vision/3D_image_classification/
        # Load best weights
        prediction_scores = []
        # self.model.load_weights("3d_attention_classification.h5")

        prediction = self.model.predict(
            np.expand_dims(self.validation_data[prediction_index],
                           axis=0))[0]

        for key in self.onehot_to_label.keys():
            index = key.index(1)
            prediction_scores.append((prediction[index],
                                      self.onehot_to_label[key], ))

        print("\nIndex: {0}".format(prediction_index))
        for score in prediction_scores:
            print(
                "This model is {0:.2f} percent confident that label is {1}"
                .format((100 * score[0]), score[1])
            )
        print("The actual label is: " +
              self.onehot_to_label[self.validation_labels[prediction_index]])

    # Useful to check for any data that the model predicts inaccurately
    def check_prediction_for_index(self, prediction_index, verbose=False):
        # self.model.load_weights("3d_attention_classification.h5")

        prediction = self.model.predict(
            np.expand_dims(self.validation_data[prediction_index],
                           axis=0))[0]

        predicted_label = [0] * len(self.onehot_to_label.keys())
        maximum = np.max(prediction)
        hot_index = np.where(prediction == maximum)[0][0]

        predicted_label[hot_index] = 1
        actual_label = self.validation_labels[prediction_index]

        p = self.onehot_to_label[tuple(predicted_label)]
        a = self.onehot_to_label[tuple(actual_label)]
        if p != a or verbose:
            print("\nIndex: {0}\nPredicted label: {1}\nActual label: {2}"
                  .format(prediction_index, p, a))

    def get_prediction_errors_for_index(self, prediction_index):
        prediction_scores = []
        prediction = self.model.predict(
            np.expand_dims(self.validation_data[prediction_index],
                           axis=0))[0]

        for key in self.onehot_to_label.keys():
            index = key.index(1)
            prediction_scores.append((prediction[index],
                                      key,))

        actual_label = self.validation_labels[prediction_index]
        prediction_errors = []
        for score in prediction_scores:
            prediction_errors.append(
                1.0 - score[0] if score[1] == actual_label else score[0])

        return prediction_errors

    def make_predictions(self, number_of_predictions, offset=0):
        if ((number_of_predictions + offset) > len(self.validation_labels)):
            print("Not enough data available for given parameters")
            return

        for index in range(number_of_predictions):
            self.predict_for_index(index + offset)

    def check_predictions(self, number_of_predictions, offset=0,
                          verbose=False):
        if ((number_of_predictions + offset) > len(self.validation_labels)):
            print("Not enough data available for given parameters")
            return

        for index in range(number_of_predictions):
            self.check_prediction_for_index(index + offset, verbose)

    def get_average_prediction_error(self):
        confidence_differences = []
        print("Calculating prediction confidence differences:")
        label_indices = trange(len(self.validation_labels))
        for index in label_indices:
            confidence_differences.extend(
                self.get_prediction_errors_for_index(index))

        average_confidence_difference = str(np.mean(confidence_differences))
        print("Average confidence difference: " +
              average_confidence_difference)
