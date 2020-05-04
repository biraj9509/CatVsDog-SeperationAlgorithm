from tkinter import filedialog
from tkinter import *
import tkinter as tkr
import numpy as np
import pickle
import cv2
import os
from extract_feature import extract_color_histogram, extraRawPixel
root = Tk()
root.geometry("500x400")

def select_image():
    filename = filedialog.askopenfilename()
    pathlabel.config(text=filename)

    TestImage(filename)
def TestImage(image):
    dataSelf1 = []
    dataSelf2 = []
    T = tkr.Text(root, height=2, width=40)


    # load 2 models from local storage
    loaded_model1 = pickle.load(open('knnpickle_model_color', 'rb'))
    loaded_model2 = pickle.load(open('knnpickle_RawPixelfile', 'rb'))

    testImage = cv2.imread(image)
    hist = extract_color_histogram(testImage)
    hist_second = extraRawPixel(testImage)
    dataSelf1.append(hist)
    label1 = loaded_model1.predict(dataSelf1)
    dataSelf2.append(hist_second)
    label2 = loaded_model2.predict(dataSelf2)
    if label1[0] == 0:
    #if label1[0] == 0 and label2[0] == 0:
        T.insert(tkr.END, "The given Input Image is predicted as a\nCat\n")
        print("the given input is predicted to be cat")
    if label1[0] == 1:
    #if label1[0] == 1 and label2[0] == 1:
        T.insert(tkr.END, "The given Input Image is predicted as a\nDog\n")
        print("The given Image is predicted to be dog")
    # if (label1[0] == 0 and label2[0] == 1) or (label1[0] == 1 and label2[0] == 0):
    #     T.insert(tkr.END, "Not defined by the current model\n")
    #     print("N/A")

    T.pack()
def runTrainModelRaw():
    os.system('python knn_train_rawPixel.py --dataset CatAndDog --model CatvsDog.model --labelbin knn.pickle')

def runTrainModelColor():
    os.system('python knn_train_color.py --dataset CatAndDog --model CatvsDog.model --labelbin knn.pickle')


browsebutton = Button(root, text="Browse", command=select_image)
browsebutton.pack()
trainModelColor = Button(root, text="Train Model With Raw pixel", command=runTrainModelRaw)
trainModelColor.pack()
trainModelRaw = Button(root, text="Train Model With Color Pixel", command=runTrainModelColor)
trainModelRaw.pack()

pathlabel = Label(root)
pathlabel.pack()

mainloop()