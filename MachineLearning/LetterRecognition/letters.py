import tensorflow as tf
import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import win_unicode_console

#raw print error because of python otherwise
win_unicode_console.enable()

#Train folder has 300 images for each letter
#Test folder has 100 images for each letter

folders = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
root = "."
trainFolder = "/lettersTrain"
testFolder = "/lettersTest"

# create a vector where the for A the first element is 1, the rest 0
# for B, the second is 1, the rest 0
# and so on
# works like one_hot
def getOneHot(alphabet):
    if(alphabet == "A"):
        return np.eye(10, dtype=np.float32)[0]
    if(alphabet == "B"):
        return np.eye(10, dtype=np.float32)[1]
    if(alphabet == "C"):
        return np.eye(10, dtype=np.float32)[2]
    if(alphabet == "D"):
        return np.eye(10, dtype=np.float32)[3]    
    if(alphabet == "E"):
        return np.eye(10, dtype=np.float32)[4]
    if(alphabet == "F"):
        return np.eye(10, dtype=np.float32)[5]
    if(alphabet == "G"):
        return np.eye(10, dtype=np.float32)[6]
    if(alphabet == "H"):
        return np.eye(10, dtype=np.float32)[7]
    if(alphabet == "I"):
        return np.eye(10, dtype=np.float32)[8]
    if(alphabet == "J"):
        return np.eye(10, dtype=np.float32)[9]		
   
# returns 2 arrays, the first contains the names of all images, the second the corresponding labels 
# labels are the path because we will use the path in batch function to read the images
def getListOfImages(fromFolder):
    global folders
    global root
	
    allImagesArray = np.array([], dtype=np.str)
    allImagesLabelsArray = np.array([], dtype=np.str)
	
    for folder in folders:
        print("Loading Image Name of ", folder)
        currentAlphabetFolder = root+fromFolder+"/"+folder+"/"
        imagesName = os.listdir(currentAlphabetFolder)
        allImagesArray = np.append(allImagesArray, imagesName) #append all names of images (feature)
        print("Nr. of images: ", len(imagesName))
        for i in range(0, len(imagesName)):
            allImagesLabelsArray = np.append(allImagesLabelsArray, currentAlphabetFolder) #append name of folder to each image (labels)
    return allImagesArray, allImagesLabelsArray

#returns randomized vector of (names+labels)
def shuffleImagesPath(imagesArray, labelsArray):
    print("Size to shuffle: ", len(imagesArray))
    for i in range(0, 100000):
		#random indexes
        randomIndex1 = randint(0, len(imagesArray)-1)
        randomIndex2 = randint(0, len(imagesArray)-1)
		#switch name and labels
        imagesArray[randomIndex1], imagesArray[randomIndex2] = imagesArray[randomIndex2], imagesArray[randomIndex1]
        labelsArray[randomIndex1], labelsArray[randomIndex2] = labelsArray[randomIndex2], labelsArray[randomIndex1]
    print("Shuffling done")
    return imagesArray, labelsArray

#after each batch the value remains because it is a global variable
batchIndex = 0
imagesArray, labelsArray = getListOfImages(trainFolder)
imagesArray, labelsArray = shuffleImagesPath(imagesArray, labelsArray)

#get images in batches for training
def getBatchOfLetterImages(batchSize):
    global batchIndex
    global imagesArray
    global labelsArray
	
	# features are pixels, 784 features
	# labels are 10-vectors containing only one 1, which represents the letter
	# ex: first element is 1 for A, second for B, third for C...
    features = np.ndarray(shape=(0, 784), dtype=np.float32)
    labels = np.ndarray(shape=(0, 10), dtype=np.float32)
	
    with tf.Session() as sess:
        for i in range(batchIndex, len(imagesArray)):
            pathToImage = labelsArray[i]+imagesArray[i]
            lastIndexOfSlash = pathToImage.rfind("/")
            folder = pathToImage[lastIndexOfSlash - 1] 
			
            imageContents = tf.read_file(str(pathToImage))
            image = tf.image.decode_png(imageContents, dtype=tf.uint8)
            resized_image = tf.image.resize_images(image, [28, 28]) 
            imarray = resized_image.eval()
            imarray = imarray.reshape(784)
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getOneHot(folder)], dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            features = np.append(features, appendingImageArray, axis=0)
            if(len(labels) >= batchSize):
                batchIndex=batchIndex+batchSize
                return labels, features




tf.reset_default_graph()

# Network Parameters (784-100-100-100-10)
n_input = 784
hidden_layer_neurons = 100
hidden_layer2_neurons = 100
hidden_layer3_neurons = 100
n_classes = 10

# training parameters
learning_rate = 0.02
epochs = 5
batch_size = 300
nrBatches = 300*10//batch_size

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
cost = tf.reduce_mean(cost + 1*regularizers);

trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define the Test model and accuracy
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



	
def getTestArrays(testImages, testLabels):
    dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
    labels = np.ndarray(shape=(0, 10), dtype=np.float32)
    with tf.Session() as sess:
        for i in range(0, len(testImages)):
            pathToImage = testLabels[i]+testImages[i]
            lastIndexOfSlash = pathToImage.rfind("/")
            folder = pathToImage[lastIndexOfSlash - 1] 
			
            imageContents = tf.read_file(str(pathToImage))
            image = tf.image.decode_png(imageContents, dtype=tf.uint8)
            resized_image = tf.image.resize_images(image, [28, 28]) 
            imarray = resized_image.eval()
            imarray = imarray.reshape(784)
            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getOneHot(folder)], dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            dataset = np.append(dataset, appendingImageArray, axis=0)
        return dataset, labels 
	
testImages, testLabels = getListOfImages(testFolder)
shufTestData, shufTestLabel = shuffleImagesPath(testImages,testLabels)
testData, testDataLabel  = getTestArrays(shufTestData, shufTestLabel)
	
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(0, epochs):
        batchIndex=0
        for j in range(0, 5):
            batchY, batchX = getBatchOfLetterImages(batch_size)
            opt = session.run(trainer, feed_dict={x: batchX, y: batchY})
            loss, acc = session.run([cost, accuracy], feed_dict={x: batchX, y: batchY})
            print("Iteration: " + str(j+1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Epoch: " + str(i+1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print("Test accuracy: ", accuracy.eval(feed_dict={x: testData, y: testDataLabel}))

    imageContents1 = tf.read_file(str("pred.png"))
    image1 = tf.image.decode_png(imageContents1, dtype=tf.uint8)
    resized_image1 = tf.image.resize_images(image1, [28, 28]) 
    imarray1 = resized_image1.eval()
    imarray1 = imarray1.reshape(784)
	
    imageContents2 = tf.read_file(str("pred2.png"))
    image2 = tf.image.decode_png(imageContents2, dtype=tf.uint8)
    resized_image2 = tf.image.resize_images(image2, [28, 28]) 
    imarray2 = resized_image2.eval()
    imarray2 = imarray2.reshape(784)
	
    feed_dict = {x: [imarray1]}
    classification = session.run(output_layer, feed_dict)
    print(classification)
	
    feed_dict = {x: [imarray2]}
    classification = session.run(output_layer, feed_dict)
    print(classification)

    savedPath = saver.save(session, "./model.ckpt")
    #print("Model saved at: " ,savedPath)
