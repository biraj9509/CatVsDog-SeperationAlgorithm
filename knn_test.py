import numpy as np
import pickle
import cv2
from extract_feature import extract_color_histogram

dataSelf = []
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
print(loaded_model)
testImage = cv2.imread('1.jpg')
hist = extract_color_histogram(testImage)
dataSelf.append(hist)
abc = loaded_model.predict(dataSelf)
print(abc)
