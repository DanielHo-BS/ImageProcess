import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
from datetime import datetime

# Convolvtion 
def convolve2D(image, kernel):
    kernel_size = kernel.shape[0]
    h ,w = image.shape
    new_image = np.zeros((h,w))
    # Padding
    image = np.pad(image, (kernel_size-1)//2)
    # Sum
    for i in range(h):
        for j in range(w):
            new_image[i,j] = (image[i:i+kernel_size,j:j+kernel_size] * kernel).sum()
    return new_image

# Max pooling (2*2)
def max_pool(img):
    ds_img = np.zeros((img.shape[0]//2,img.shape[1]//2))
    strides = 2
    # Max
    for i in range(ds_img.shape[0]):
        for j in range(ds_img.shape[1]):
            ds_img[i,j]=img[strides*i+0:strides*i+2,strides*j+0:strides*j+2].max()           
    return ds_img

# Feature extraction
def featureExtraction(input, filter1, filter2): 
    output = np.zeros((9,200))
    # Simplify CNN
    for i in range(200):
        # Leyer 1: Convolve2D
        conv1 = convolve2D(input[:,:,i],filter1)
        conv2 = convolve2D(input[:,:,i],filter2)
        # Leyer 1: Max pooling
        conv1 = max_pool(conv1)
        conv2= max_pool(conv2)
        featuremap = np.stack((conv1,conv2),axis=0)
        # Leyer 2: Convolve2D
        conv1 = convolve2D(featuremap[0],filter1)
        conv1 = convolve2D(conv1,filter2)
        conv2 = convolve2D(featuremap[1],filter1)
        conv2 = convolve2D(conv2,filter2)
        # Leyer 2: Max pooling
        conv1 = max_pool(conv1)
        conv2= max_pool(conv2)
        featuremap = np.stack((conv1,conv2),axis=0)
        # Flatten
        output[0:8,i] = featuremap.flatten()
        # Add bias
        output[8,i] = 1
    return output.T

# Linear regression
def linearRegression(X,Y):
    return (np.linalg.inv(X.T @ X)) @ X.T @ Y

# Prediction
def prediction(X,Y,A):
    Yp = np.zeros((200,1))
    Yp[X.dot(A) >= 0.5] = 1
    # Confuse matrix
    confuse_matrix = np.zeros((2,2))
    confuse_matrix[0,0] = np.sum([Yp[0:100] == Y[0:100]])
    confuse_matrix[1,0] = np.sum([Yp[0:100] != Y[0:100]])
    confuse_matrix[1,1] = np.sum([Yp[100:200] == Y[100:200]])
    confuse_matrix[0,1] = np.sum([Yp[100:200] != Y[100:200]])
    print("confuse_matrix: \n", confuse_matrix,"\n")
    return Yp

# Show the result with random sampling
def result(img, Y, Yp):
    # Make label
    GT = np.zeros((200,1))
    Pred = np.zeros((200,1))
    GT[Y == 0 ] = a
    GT[Y == 1 ] = b
    Pred[Yp == 0 ] = a
    Pred[Yp == 1 ] = b
    # Setting random seed
    random.seed(datetime.now())
    index = random.sample(list(range(200)),15)
    # Subplot with 3*5 windows
    plt.figure()
    for i in range(15):
            plt.subplot(3,5,i+1)
            plt.imshow(img[:,:,index[i]],"gray")
            plt.axis('off')
            if GT[index[i]] == Pred[index[i]]:
                    color="blue"
            else:
                    color = "red"
            plt.title(str(int(Pred[index[i]]))+" ("+str(index[i])+")",color=color)
    plt.show()

if __name__ == '__main__':

    # Load the data
    imgs_in = np.zeros((28,28,200))
    imgs = np.zeros((8,8,200))
    a = int(input("a: (0~9)"))
    b = int(input("b: (0~9)"))
    for i in range(100):
        imgs_in[:,:,i] = cv.imread("train1000/"+str(100*a+i+1)+".png",cv.IMREAD_GRAYSCALE)
        imgs_in[:,:,i+100] = cv.imread("train1000/"+str(100*b+i+1)+".png",cv.IMREAD_GRAYSCALE)
        imgs[:,:,i] = cv.resize(imgs_in[:,:,i], (8,8))
        imgs[:,:,i+100] = cv.resize(imgs_in[:,:,i+100], (8,8))

    # Create filter1 & filter2
    filter1 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
    filter2 = filter1.T

    # Feature Extraction
    feature = featureExtraction(imgs,filter1, filter2)

    # Linear regression
    target = np.concatenate((np.zeros((100,1)),np.ones((100,1))))
    coef = linearRegression(feature,target)

    # Prediction
    pred = prediction(feature, target,coef)

    # Result
    result(imgs_in, target, pred)