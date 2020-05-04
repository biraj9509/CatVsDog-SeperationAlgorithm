# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from extract_feature import extract_color_histogram, extraRawPixel
import numpy as np
import argparse
import imutils
import time
import pickle
import cv2
import os




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True,
  help="path to output label binarizer")
args = vars(ap.parse_args())


#Get path of image from args to load the image
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the array
data = []
labels = []
for (i, imagePath) in enumerate(imagePaths):
    # load image and read its matrix value
    image = cv2.imread(imagePath)
    # splits the image name of dataset with delimiter '.' and only keeps the name
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract a color histogram from the image
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("Constructing training/testing split...")

#changes the label of image into binary form, 0 for cat and 1 for dog
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainData, testDataFoo, trainLabels, testLabelsFoo) = train_test_split(
    data, labels, test_size=0.40, random_state=42)
(validationData, testData, trainLabels, testLabels) = train_test_split(
    testDataFoo, testDataFoo, test_size=0.50, random_state=42)
#print(testData)
#print(labels)
# construct the set of hyperparameters to tune
params = {"n_neighbors": np.arange(1, 31, 2),
          "metric": ["euclidean", "cityblock"]}

# tune the knn classifier parameter using cross-validated grid search for better accuracy
print("Tuning hyperparameters: ")
model = KNeighborsClassifier(n_jobs=args["jobs"])
finalModel = GridSearchCV(model, params)
start = time.time()
# fits the model in KNN and np.ravel changes the structure of trainLabel into accepted format
finalModel.fit(trainData, np.ravel(trainLabels))


# evaluate the best grid searched model on the testing data
per_score = finalModel.score(testData, testLabels)
print("Final model accuracy : {:.2f}%".format(per_score * 100))
print("Using cross validated grid Search it took {:.2f} seconds".format(
    time.time() - start))
# tune the hyperparameters again using  randomized search
finalModel = RandomizedSearchCV(model, params)
start = time.time()
finalModel.fit(trainData, np.ravel(trainLabels))


# evaluate the best randomized searched model on the testing
# data
print("Using Randomized Search it took {:.2f} seconds".format(
    time.time() - start))
accuracy = finalModel.score(testData, testLabels)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# exports the trained model into local storage for future use
os.chdir( 'model' )
knnPickle = open(args["model"], 'wb')

# source, destination
pickle.dump(finalModel, knnPickle)
os.chdir('../')