# Import useful libraries for data analysis
import numpy as np
import pandas as pd

# Import useful libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import Keras Modules (Neural Network)
from keras.preprocessing.image import ImageDataGenerator

# Import from other files in project
from constants import *

def generate_subsets():
    """
    Function that will take the DATA path and will generate the 3 subsets of the dataset for TRAIN, VALIDATION and TEST
    :return:
    """
    # Data Augmentation
    # ImageDataGenerator (Keras) =  data augmentation, technique to expand the training dataset size
    #                               by creating a modified version of the original image
    #                               which can improve model performance and the ability to generalize.
    # Parameters:
    # - rotation_range: Degree range for random rotations. We choose 360 degrees since the product is a round object.
    # - width_shift_range: Fraction range of the total width to be shifted.
    # - height_shift_range: Fraction range of the total height to be shifted.
    # - shear_range: Degree range for random shear in a counter-clockwise direction.
    # - zoom_range: Fraction range for random zoom.
    # - horizontal_flip and vertical_flip are set to True for randomly flip image horizontally and vertically.
    # - brightness_range: Fraction range for picking a brightness shift value.
    # - rescale: Rescale the pixel values to be in range 0 and 1.
    # - validation_split: Reserve 20% of the training data for validation, and the rest 80% for model fitting
    train_generator = ImageDataGenerator(rotation_range=360,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         shear_range=0.05,
                                         zoom_range=0.05,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         brightness_range=[0.75, 1.25],
                                         rescale=1. / 255,
                                         validation_split=0.2)

    # flow_from_directory parameters:
    # color_mode = "grayscale": Treat our image with only one channel color.
    # class_mode and classes define the target class of our problem.
    # In this case, we denote the defect class as positive (1), and ok as a negative (0) class.
    # shuffle = True to make sure the model learns the defect and ok images alternately.
    train_dataset = train_generator.flow_from_directory(directory=DATA_DIRECTORY + "train",
                                                        subset="training", **generator_args)
    validation_dataset = train_generator.flow_from_directory(directory=DATA_DIRECTORY + "train",
                                                             subset="validation", **generator_args)

    # No data augmentation on the test data
    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_dataset = test_generator.flow_from_directory(directory=DATA_DIRECTORY + "test",
                                                      **generator_args)

    return train_dataset, validation_dataset, test_dataset


def plot_subset_percetanges(train_dataset, validation_dataset, test_dataset):
    """
    Generate plot of the counts+percentages of the subsets
    :param train_dataset:
    :param validation_dataset:
    :param test_dataset:
    :return:
    """
    image_data = []
    for dataset, type in zip([train_dataset, validation_dataset, test_dataset], ["train", "validation", "test"]):
        for name in dataset.filenames:
            image_data.append({"data": type,
                               "class": name.split('\\')[0],
                               "filename": name.split('\\')[1]})

    image_df = pd.DataFrame(image_data)
    data_crosstab = pd.crosstab(index=image_df["data"],
                                columns=image_df["class"],
                                margins=True,
                                margins_name="Total")
    print(data_crosstab)

    total_image = data_crosstab.iloc[-1, -1]
    ax = data_crosstab.iloc[:-1, :-1].T.plot(kind="bar", stacked=True, rot=0)

    percent_val = []

    for rect in ax.patches:
        height = rect.get_height()
        width = rect.get_width()
        percent = 100 * height / total_image

        ax.text(rect.get_x() + width - 0.25,
                rect.get_y() + height / 2,
                int(height),
                ha='center',
                va='center',
                color="white",
                fontsize=10)

        ax.text(rect.get_x() + width + 0.01,
                rect.get_y() + height / 2,
                "{:.2f}%".format(percent),
                ha='left',
                va='center',
                color="black",
                fontsize=10)

        percent_val.append(percent)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    percent_def = sum(percent_val[::2])
    ax.set_xticklabels(["def_front\n({:.2f} %)".format(percent_def), "ok_front\n({:.2f} %)".format(100 - percent_def)])
    plt.title("Data Segmentation", fontsize=15, fontweight="bold")
    plt.savefig("./plots/data_segmentation")
    plt.show()


def plot_data_batch(title, dataset):
    images, labels = next(iter(dataset))
    images = images.reshape(BATCH_SIZE, *DATA_SIZE)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(mapping_class[label], size=20)

    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle(title, size=30, y=0.99, fontweight="bold")
    plt.show()

    return images


def plot_training_accuracy(fitted_model):
    # summarize history for accuracy
    plt.plot(fitted_model.history['accuracy'])
    plt.plot(fitted_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("./plots/training_accuracy")
    plt.show()


def plot_training_loss(fitted_model):
    # summarize history for loss
    plt.plot(fitted_model.history['loss'])
    plt.plot(fitted_model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("./plots/training_loss")
    plt.show()


def plot_training_evaluation(fitted_model):
    plt.subplots(figsize=(8, 6))
    sns.lineplot(data=pd.DataFrame(fitted_model.history,
                                   index=range(1, 1 + len(fitted_model.epoch))))
    plt.title("Training Evaluation", fontweight="bold", fontsize=20)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Metrics")

    plt.legend(labels=['validation loss', 'validation accuracy', 'train loss', 'train accuracy'])
    plt.savefig("./plots/training_evaluation")
    plt.show()


def plot_test_random_results(test_dataset, best_model):
    images, labels = next(iter(test_dataset))
    images = images.reshape(BATCH_SIZE, *DATA_SIZE)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap="gray")
        true_label = mapping_class[label]

        [[pred_prob]] = best_model.predict(img.reshape(1, *DATA_SIZE, -1))
        pred_label = mapping_class[int(pred_prob >= PRED_THRESHOLD)]

        prob_class = 100 * pred_prob if pred_label == "defect" else 100 * (1 - pred_prob)

        ax.set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=18)
        ax.set_xlabel(f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {(prob_class):.2f}%",
                      fontweight="bold", fontsize=15,
                      color="blue" if true_label == pred_label else "red")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle("TRUE VS PREDICTED LABEL FOR 16 RANDOM TEST IMAGES", size=30, y=0.99, fontweight="bold")
    plt.savefig("./plots/16random_results")
    plt.show()

def plot_test_missed_results(test_dataset, y_true_class, y_pred_class, y_pred_prob):
    misclassify_pred = np.nonzero(y_pred_class != y_true_class)[0]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    test_indexes = test_dataset.index_array
    # print(test_indexes[misclassify_pred])
    # for i in range(0, len(misclassify_pred)):
    #     test_index = test_indexes[misclassify_pred[i]]
    #     print(y_true_class[test_index])
    #     print(y_pred_class[test_index])

    i = 0
    for ax, batch_num, image_num in zip(axes.flat, test_indexes[misclassify_pred] // BATCH_SIZE, test_indexes[misclassify_pred] % BATCH_SIZE):
        images, labels = test_dataset[batch_num]
        img = images[image_num]
        ax.imshow(img.reshape(*DATA_SIZE), cmap="gray")

        # true_label = mapping_class[labels[image_num]]
        # [[pred_prob]] = best_model.predict(img.reshape(1, *IMAGE_SIZE, -1))
        # pred_label = mapping_class[int(pred_prob >= THRESHOLD)]

        true_label = mapping_class[y_true_class[misclassify_pred[i]]]
        [pred_prob] = y_pred_prob[misclassify_pred[i]]
        pred_label = mapping_class[int(pred_prob >= PRED_THRESHOLD)]

        prob_class = 100 * pred_prob if pred_label == "defect" else 100 * (1 - pred_prob)

        ax.set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=18)
        ax.set_xlabel(f"PREDICTED LABEL: {pred_label}\nProb({pred_label}) = {(prob_class):.2f}%",
                      fontweight="bold", fontsize=15,
                      color="blue" if true_label == pred_label else "red")

        ax.set_xticks([])
        ax.set_yticks([])

        i += 1

    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle(f"MISCLASSIFIED TEST IMAGES ({len(misclassify_pred)} out of {len(y_true_class)})", size=20, y=0.99, fontweight="bold")
    plt.savefig("./plots/missed_results")
    plt.show()
