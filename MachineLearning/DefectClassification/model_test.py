# Import Keras Modules (for Neural Network)
from keras.engine.saving import load_model

# Import useful libraries for evaluation
import numpy as np
from numpy.random._generator import default_rng
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import from other files in project
from dataset import *


# Accuracy: 99.58%
# Class type:        0(ok)         1(defect)
# Precision:        99.24%          99.78%
# Recall:           99.62%          99.56%
# F1 score:         99.43%          99.67%
def main():
    _, _, test_dataset = generate_subsets()

    # LOAD MODEL TRAINED (ONLY BEST IS SAVED)
    best_model = load_model("./cnn_model.hdf5")

    predicted_probabilities = best_model.predict_generator(generator=test_dataset,
                                               verbose=1)

    # create column array with predicted labels
    predicted_labels = (predicted_probabilities >= PREDICTION_THRESHOLD).reshape(-1, )
    true_labels = test_dataset.classes[test_dataset.index_array]

    print(
        pd.DataFrame(
            confusion_matrix(true_labels, predicted_labels),
            index=[["Actual", "Actual"], ["ok", "defect"]],
            columns=[["Predicted", "Predicted"], ["ok", "defect"]],
        ))

    print(classification_report(true_labels, predicted_labels, digits=4))

    test_indexes = test_dataset.index_array
    rng = default_rng()
    random_indexes = rng.choice(len(test_indexes), size=16, replace=False)
    plot_results("random", test_dataset, random_indexes, true_labels, predicted_probabilities)

    misclassified_indexes = np.nonzero(predicted_labels != true_labels)[0]
    plot_results("missed", test_dataset, misclassified_indexes, true_labels, predicted_probabilities)

if __name__ == "__main__":
    main()
