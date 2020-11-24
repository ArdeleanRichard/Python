# DATA_DIRECTORY: The directory where the image data is stored.
DATA_DIRECTORY = "./casting_data/"
# DATA_SIZE: The dimension of the image (300 px by 300 px).
DATA_SIZE = (300, 300)
# BATCH_SIZE: Number of images that will be loaded and trained at one time.
BATCH_SIZE = 64
# SEED_NUMBER: Ensure reproducibility.
SEED = 100
# For each epoch, batch_size  ×  steps_per_epoch images will be fed into our CNN architecture.
# In this case, we specify the steps_per_epoch to be 150
# so for each epoch 64 * 150 = 9600 augmented images from the training dataset will be fed.
STEPS_PER_EPOCH = 150
NR_EPOCHS = 25
# We let the model train for 25 epochs.
# PRED_THRESHOLD: Threshold used to separate the classes.
# If the probability is greater or equal, then it will be classified as defect, otherwise ok
PREDICTION_THRESHOLD = 0.5
LABEL_DICTIONARY = {0: "ok", 1: "defect"}



# GENERATE SUB-DATASET IMAGES
# flow_from_directory parameters:
# color_mode = "grayscale": Treat our image with only one channel color.
# class_mode and classes define the target class of our problem.
# In this case, we denote the defect class as positive (1), and ok as a negative (0) class.
# shuffle = True to make sure the model learns the defect and ok images alternately.
ARGS_GENERATOR = dict(target_size=DATA_SIZE,
                      color_mode="grayscale",
                      batch_size=BATCH_SIZE,
                      class_mode="binary",
                      classes={"ok_front": 0, "def_front": 1},
                      shuffle=True,
                      seed=SEED)
