# Steps of the process:
#   1. Load the images and apply the data augmentation technique (dataset.py)
#   2. Training with validation: (model_train.py)
#           define the architecture
#           compile the model
#           model fitting and evaluation
#   3. Testing on unseen images (model_test.py)


# Import libraries for visualization
import seaborn as sns
sns.set()

# Import Keras Modules (for Neural Network)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# Import from other files in project
from dataset import *


def create_model():
    # NN ARCHITECTURE
    # First convolutional layer: consists of 32 filters with kernel_size matrix 3 by 3.
    #                       Using 2-pixel strides at a time, reduce the image size by half.
    # First pooling layer: Using max-pooling matrix 2 by 2 (pool_size)
    #                       and 2-pixel strides at a time further reduce the image size by half.
    # Second convolutional layer: Just like the first convolutional layer but with 16 filters only.
    # Second pooling layer: Same as the first pooling layer.
    # Flattening: Convert two-dimensional pixel values into one dimension,
    #                       so that it is ready to be fed into the fully-connected layer.
    # First dense layer + Dropout: consists of 128 units and 1 bias unit.
    #                       Dropout of rate 20% is used to prevent overfitting.
    # Second dense layer + Dropout: consists of 64 units and 1 bias unit.
    #                       Dropout of rate 20% is also used to prevent overfitting.
    # Output layer: consists of only one unit and activation is a sigmoid function
    #                       to convert the scores into a probability of an image being defect.
    model = Sequential(
        [
            # First convolutional layer
            Conv2D(filters=32, kernel_size=3, strides=2, activation="relu", input_shape=DATA_SIZE + (1,)),

            # First pooling layer
            MaxPooling2D(pool_size=2, strides=2),

            # Second convolutional layer
            Conv2D(filters=16, kernel_size=3, strides=2, activation="relu"),

            # Second pooling layer
            MaxPooling2D(pool_size=2, strides=2),

            # Flattening
            Flatten(),

            # Fully-connected layer
            Dense(128, activation="relu"),
            Dropout(rate=0.2),

            # Fully-connected layer
            Dense(64, activation="relu"),
            Dropout(rate=0.2),

            # Fully-connected output layer (1 neuron either 0 for non-defect or 1 for defect)
            Dense(1, activation="sigmoid")
        ]
    )

    return model


def train_model(model, train_dataset, validation_dataset):
    # The loss function chosen to be optimized for your model is calculated at the end of each epoch.
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model.
    # Early stopping is a method that allows you to specify an arbitrary large number of training epochs
    # and stop training once the model performance stops improving on a hold out validation dataset.
    # - seek a minimum for validation loss
    # - model may hit a plateau of no improvement or even get slightly worse before getting much better
    #       -add delay to the trigger in terms of the number of epochs (patience = 2)
    # - other early stopping parameters:
    #       min_delta = considers improvement by units (accuracy uses percentage)
    #       baseline = threshold once achieved, it can stop
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # the best model will be automatically saved if the current val_loss is lower than the previous one.
    checkpoint = ModelCheckpoint("cnn_model_trial5.hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 monitor="val_loss")

    # Callbacks provide a way to execute code and interact with the training model process automatically.
    # the model is evaluated on the validation dataset at the end of each training epoch.
    fitted_model = model.fit_generator(generator=train_dataset,
                                       validation_data=validation_dataset,
                                       steps_per_epoch=STEPS_PER_EPOCH,
                                       epochs=NR_EPOCHS,
                                       validation_steps=STEPS_PER_EPOCH,
                                       callbacks=[checkpoint, early_stop],
                                       verbose=1)

    return fitted_model


def main():
    # Generate train, validation, test datasets
    train_dataset, validation_dataset, test_dataset = generate_subsets()

    # Plot data segmentation of the datasets
    plot_subset_percentages(train_dataset, validation_dataset, test_dataset)

    # Plot the first batches of the training and the testing datasets
    batch_number = 1
    plot_data_batch(f"TRAINING BATCH {batch_number}\n(AUGMENTED DATA)", train_dataset, batch_number=batch_number)
    plot_data_batch(f"TESTING BATCH {batch_number}\n(NON-AUGMENTED DATA)", test_dataset, batch_number=batch_number)

    model = create_model()

    model.summary()

    trained_model = train_model(model, train_dataset, validation_dataset)

    # list all data in history
    # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    print(trained_model.history.keys())

    plot_training_evaluation(trained_model)
    plot_training_metric(trained_model, "accuracy")
    plot_training_metric(trained_model, "loss")


if __name__ == "__main__":
    main()
