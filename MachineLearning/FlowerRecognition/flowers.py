import tensorflow as tf
import pandas as pd
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt
import win_unicode_console

win_unicode_console.enable()

dir_path = os.path.dirname(os.path.realpath(__file__))
img_dir = dir_path + "/all_flower_images"
csv_file = dir_path + "/flowers_ds.csv"

def readCSV():
    data = pd.read_csv(csv_file, header = 1, names=['file', 'label'])

    # shuffle all
    data = data.sample(frac = 1)

    # reshuffle and split
    test_img = data.sample(frac=0.03)
    train_img = data.drop(test_img.index)

    return train_img, test_img


train_set, test_set = readCSV()

train_images = train_set["file"].tolist()
train_labels = train_set["label"].tolist()

test_images = test_set["file"].tolist()
test_labels = test_set["label"].tolist()

def getNumber(alphabet):
    if(alphabet == 0):
        return np.eye(5, dtype=np.float32)[0]
    if(alphabet == 1):
        return np.eye(5, dtype=np.float32)[1]
    if(alphabet == 2):
        return np.eye(5, dtype=np.float32)[2]
    if(alphabet == 3):
        return np.eye(5, dtype=np.float32)[3]    
    if(alphabet == 4):
        return np.eye(5, dtype=np.float32)[4]

		
startIndexOfBatch = 0
def getBatch(batchSize):
    global startIndexOfBatch
    global train_images
    global train_labels
	
    features = np.ndarray(shape=(0, 2352), dtype=np.float32)
    labels = np.ndarray(shape=(0, 5), dtype=np.float32)
	
    with tf.Session() as sess:
        for i in range(startIndexOfBatch, len(train_images)):
            pathToImage = dir_path+"/all_flower_images/"+train_images[i]

            imageContents = tf.read_file(str(pathToImage))
            image = tf.image.decode_jpeg(imageContents, channels=3)
            resized_image = tf.image.resize_images(image, [28, 28]) 
            imarray = resized_image.eval()
            imarray = imarray.reshape(1, -1)
            appendingImageArray = np.array(imarray, dtype=np.float32)
            appendingNumberLabel = np.array(np.transpose(np.reshape(getNumber(train_labels[startIndexOfBatch]),(getNumber(train_labels[startIndexOfBatch]).shape[0],1))), dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            features = np.append(features, appendingImageArray, axis=0)
            startIndexOfBatch=startIndexOfBatch+1
            if(len(features) >= batchSize):
                return labels, features

tf.reset_default_graph()

n_input = 2352
hidden_layer_neurons = 100
hidden_layer2_neurons = 50
hidden_layer3_neurons = 25
n_classes = 5

# training parameters
learning_rate = 0.2
epochs = 5
batch_size = 334

# parameters
# x and y placeholders
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create weights and biases that will be used in the neural network
w1 = tf.Variable(tf.random_normal([n_input, hidden_layer_neurons]))
w2 = tf.Variable(tf.random_normal([hidden_layer_neurons, hidden_layer2_neurons]))
w3 = tf.Variable(tf.random_normal([hidden_layer2_neurons, hidden_layer3_neurons]))
w4 = tf.Variable(tf.random_normal([hidden_layer3_neurons, n_classes]))

b1 = tf.Variable(tf.random_normal([hidden_layer_neurons]))
b2 = tf.Variable(tf.random_normal([hidden_layer2_neurons]))
b3 = tf.Variable(tf.random_normal([hidden_layer3_neurons]))
b4 = tf.Variable(tf.random_normal([n_classes]))

# The multilayer perceptron model
hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer, w2), b2))
hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, w3), b3))
output_layer = tf.add(tf.matmul(hidden_layer3, w4), b4)



# Cost funcition and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))

regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
cost = tf.reduce_mean(cost + 1*regularizers);

trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define the Test model and accuracy
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def getTestArrays():
    global test_images
    global test_labels
	
    dataset = np.ndarray(shape=(0, 2352), dtype=np.float32)
    labels = np.ndarray(shape=(0, 5), dtype=np.float32)
    with tf.Session() as sess:
        for i in range(0, len(test_images)):
            pathToImage = dir_path+"/all_flower_images/"+test_images[i]
			
            imageContents = tf.read_file(str(pathToImage))
            image = tf.image.decode_jpeg(imageContents, channels=3)
            resized_image = tf.image.resize_images(image, [28, 28]) 
            imarray = resized_image.eval()
            imarray = imarray.reshape(1, -1)
            appendingImageArray = np.array(imarray, dtype=np.float32)
            appendingNumberLabel = np.array(np.transpose(np.reshape(getNumber(test_labels[i]),(getNumber(test_labels[i]).shape[0],1))), dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            dataset = np.append(dataset, appendingImageArray, axis=0)
        return dataset, labels 
	
testData, testDataLabel  = getTestArrays()

with tf.Session() as session:
    session.run( tf.global_variables_initializer())

    print (len(train_images))
    print (len(test_images))
    for i in range(0, epochs):
        startIndexOfBatch=0
        for j in range(0, 11):
            batchY, batchX = getBatch(batch_size)
            opt = session.run(trainer, feed_dict={x: batchX, y: batchY})
            loss, acc = session.run([cost, accuracy], feed_dict={x: batchX, y: batchY})
            print("Iteration: " + str(j+1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Epoch: " + str(i+1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print("Test accuracy: ", accuracy.eval(feed_dict={x: testData, y: testDataLabel}))





