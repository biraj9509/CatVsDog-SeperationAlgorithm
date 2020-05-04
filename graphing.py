#find auc
import numpy as np
from sklearn import metrics
from extract_feature import extract_color_histogram, extraRawPixel
import argparse
import pickle
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

data = []
labels = []
splittedValidation = []
splittedPrediction = []
j =0
target_names = ['class 0', 'class 1']

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance")
args = vars(ap.parse_args())


# prediction = []
imagePaths = list(paths.list_images(args["dataset"]))
for (i, imagePath) in enumerate(imagePaths):
    # load image and read its matrix value
    image = cv2.imread(imagePath)
    # splits the image name of dataset with delimiter '.' and only keeps the name
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract a histogram from the image
    hist = extract_color_histogram(image)

    data.append(hist)
    labels.append(label)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainData, testDataFoo, trainLabels, testLabelsFoo) = train_test_split(
    data, labels, test_size=0.40, random_state=42)
(validationData, testData, validationLabels, testLabels) = train_test_split(
    testDataFoo, testLabelsFoo, test_size=0.50, random_state=42)

loaded_model = pickle.load(open('knnpickle_model_color', 'rb'))
#loaded_model = pickle.load(open('knnpickle_RawPixelfile', 'rb'))
prediction = loaded_model.predict(validationData)

accuracy = loaded_model.score(testData, testLabels)
accuracy_validation = loaded_model.score(validationData, validationLabels)

splittedValidation = validationLabels[:10]
splittedPrediction = prediction[:10]
print("Predicted output: ")
print(splittedPrediction)
print("True Value: ")
print(splittedValidation)
# print("Accuracy from model using color parameter: {:.2f}%".format(accuracy * 100))
# print("Accuracy from model using color parameter but with validation dataset: {:.2f}%".format(accuracy_validation * 100))
#
# print(accuracy_validation)


print(classification_report(splittedValidation, splittedPrediction, target_names=target_names))
fpr, tpr, threshold = roc_curve(validationLabels, prediction)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()


#
# fpr, tpr, thresholds = metrics.roc_curve(np.array(validationData), np.array(prediction), pos_label=2)
# metrics.auc(fpr, tpr)
#
# import numpy as np
# from sklearn import metrics
# y = np.array([1, 1, 2, 2])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
# metrics.auc(fpr, tpr)