# Import from standard libraries
from pathlib import Path

# Import useful libraries for data analysis
import numpy as np
import pandas as pd

# Import useful libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import Keras Modules (Neural Network)
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator

# Import useful libraries for evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Import from other files in project
from constants import *
from dataset import *


# Accuracy: 98.46%
# Class type:        0(ok)         1(defect)
# Precision:        99.60%          97.84%
# Recall:           96.18%          99.78%
# F1 score:         97.86%          98.80%
def main():
    _, _, test_dataset = generate_subsets()

    # LOAD MODEL TRAINED (ONLY BEST IS SAVED)
    best_model = load_model("./cnn_model.hdf5")

    y_pred_prob = best_model.predict_generator(generator=test_dataset,
                                               verbose=1)

    y_pred_class = (y_pred_prob >= PRED_THRESHOLD).reshape(-1, )
    y_true_class = test_dataset.classes[test_dataset.index_array]

    print(
        pd.DataFrame(
            confusion_matrix(y_true_class, y_pred_class),
            index=[["Actual", "Actual"], ["ok", "defect"]],
            columns=[["Predicted", "Predicted"], ["ok", "defect"]],
        ))

    print(classification_report(y_true_class, y_pred_class, digits=4))

    plot_test_random_results(test_dataset, best_model)
    plot_test_missed_results(test_dataset, y_true_class, y_pred_class, y_pred_prob)


if __name__ == "__main__":
    main()
