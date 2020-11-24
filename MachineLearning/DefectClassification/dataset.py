# Import libraries for data analysis
import pandas as pd

# Import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import Keras Modules (for Neural Network)
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
                                                        subset="training", **ARGS_GENERATOR)
    validation_dataset = train_generator.flow_from_directory(directory=DATA_DIRECTORY + "train",
                                                             subset="validation", **ARGS_GENERATOR)

    # No data augmentation on the test data
    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_dataset = test_generator.flow_from_directory(directory=DATA_DIRECTORY + "test",
                                                      **ARGS_GENERATOR)

    return train_dataset, validation_dataset, test_dataset


def plot_subset_percentages(train_dataset, validation_dataset, test_dataset):
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


def plot_data_batch(title, dataset, batch_number=1):
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))

    images, labels = dataset[batch_number]

    for ax, number in zip(axes.flat, range(0, BATCH_SIZE)):
        image = images[number]
        ax.imshow(image.reshape(*DATA_SIZE), cmap="gray")
        ax.axis("off")
        ax.set_title(LABEL_DICTIONARY[labels[number]], size=20)

    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle(title, size=30, y=0.99, fontweight="bold")
    plt.show()

    return images


def plot_training_metric(fitted_model, metric):
    # summarize history for accuracy/loss
    if metric == "accuracy" or metric == "loss":
        plt.plot(fitted_model.history[f'{metric}'])
        plt.plot(fitted_model.history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.ylabel(f'{metric}')
        plt.xlabel('Number of Epochs')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"./plots/training_{metric}")
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


def __create_plot_ax(axes, test_dataset, indexes, true_labels, predicted_probability):
    test_indexes = test_dataset.index_array

    i = 0
    for ax, batch_number, image_number in zip(axes.flat, test_indexes[indexes] // BATCH_SIZE, test_indexes[indexes] % BATCH_SIZE):
        images, labels = test_dataset[batch_number]
        image = images[image_number]

        true_label = LABEL_DICTIONARY[true_labels[indexes[i]]]
        [prediction_probability] = predicted_probability[indexes[i]]
        predicted_label = LABEL_DICTIONARY[int(prediction_probability >= PREDICTION_THRESHOLD)]

        probability = 100 * prediction_probability if predicted_label == "defect" else 100 * (1 - prediction_probability)

        ax.imshow(image.reshape(*DATA_SIZE), cmap="gray")
        ax.set_title(f"TRUE LABEL: {true_label}", fontweight="bold", fontsize=18)
        ax.set_xlabel(f"PREDICTED LABEL: {predicted_label}\nProb({predicted_label}) = {probability :.2f}%",
                      fontweight="bold", fontsize=15,
                      color="blue" if true_label == predicted_label else "red")

        ax.set_xticks([])
        ax.set_yticks([])

        i+=1


def plot_results(mode, test_dataset, indexes, true_labels, predicted_probability):
    if mode != "random" and mode != "missed":
        return
    else:
        if mode == "random":
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))

            __create_plot_ax(axes, test_dataset, indexes, true_labels, predicted_probability)

            fig.suptitle(f"{len(indexes)} RANDOM TEST IMAGES", size=30, y=0.99, fontweight="bold")
            plt.savefig(f"./plots/{len(indexes)}_random_results")

        elif mode == "missed":
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            __create_plot_ax(axes, test_dataset, indexes, true_labels, predicted_probability)

            fig.subplots_adjust(top=0.8)
            fig.suptitle(f"MISCLASSIFIED TEST IMAGES ({len(indexes)}/{len(true_labels)})", size=20, y=0.99, fontweight="bold")
            plt.savefig("./plots/missed_results")

        plt.tight_layout()
        plt.show()
