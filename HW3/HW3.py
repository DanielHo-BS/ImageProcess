import numpy as np
from cv2 import resize, imread, IMREAD_GRAYSCALE
from datetime import datetime
from matplotlib import pyplot as plt
from random import seed, sample

def convolve2D(img, kernel):
    kernel_size = kernel.shape[0]
    h ,w = img.shape
    img_new = np.zeros((h,w))
    img = np.pad(img, (kernel_size-1)//2)  # Padding img with kernel size
    for i in range(h):
        for j in range(w):
            img_new[i,j] = (img[i:i+kernel_size,
                j:j+kernel_size] * kernel).sum()   # Sum the array

    return img_new

def max_pool(img):
    img_new = np.zeros((img.shape[0]//2,img.shape[1]//2))
    strides = 2
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            img_new[i,j]=img[strides*i+0:strides*i+2,
                strides*j+0:strides*j+2].max()  # Find the maximum of array

    return img_new

def feature_extraction(input, filter1, filter2): 
    """
    Using simplify CNN to extracte feature,
    with 2 layers (CNN + MaxPool) and 1 flatten layer.
    Input size: 8*8*200
    Output size: 200*9
    Filter size: 3*3
    MaxPool size: 2*2
    """
    output = np.zeros((9,200))
    for i in range(200):
        # Leyer 1: Convolve2D + MaxPool
        conv1 = convolve2D(input[:,:,i],filter1)  # output: 8*8
        conv2 = convolve2D(input[:,:,i],filter2)
        conv1 = max_pool(conv1)  #  output: 4*4
        conv2= max_pool(conv2)
        featuremap = np.stack((conv1,conv2),axis=0)  # output: 4*4*2
        # Leyer 2: Convolve2D + MaxPool
        conv1 = convolve2D(featuremap[0],filter1)  # output: 4*4
        conv1 = convolve2D(conv1,filter2)
        conv2 = convolve2D(featuremap[1],filter1)
        conv2 = convolve2D(conv2,filter2)
        conv1 = max_pool(conv1)  # output: 2*2
        conv2= max_pool(conv2)
        featuremap = np.stack((conv1,conv2),axis=0)  # output: 2*2*2
        # Flatten layer
        output[0:8,i] = featuremap.flatten()  # from 2*2*2 to 8*1
        output[8,i] = 1  # Add bias

    return output.T  # Make output size = 200*9

def linear_regression(X,Y):
    return (np.linalg.inv(X.T @ X)) @ X.T @ Y

def prediction(X,Y,A):
    Yp = np.zeros((200,1))
    Yp[X.dot(A) >= 0.5] = 1  # Make prediction value > 0.5 = 1, others = 0
    confuse_matrix = np.zeros((2,2))  # Create confuse matrix
    confuse_matrix[0,0] = np.sum([Yp[0:100] == Y[0:100]])
    confuse_matrix[1,0] = np.sum([Yp[0:100] != Y[0:100]])
    confuse_matrix[1,1] = np.sum([Yp[100:200] == Y[100:200]])
    confuse_matrix[0,1] = np.sum([Yp[100:200] != Y[100:200]])
    print("confuse_matrix: \n", confuse_matrix,"\n")
    return Yp

def result(img, Y, Yp):
    ground_true = np.zeros((200,1))
    predict = np.zeros((200,1))
    ground_true[Y == 0 ] = NUMBER1  # Change label 0 into NUMBER1  
    ground_true[Y == 1 ] = NUMBER2  # Change label 1 into NUMBER2 
    predict[Yp == 0 ] = NUMBER1  # Change label 0 into NUMBER1
    predict[Yp == 1 ] = NUMBER2  # Change label 1 into NUMBER2
    seed(datetime.now())  # Setting random seed
    index = sample(list(range(200)),15)  # Random sample 15 index from [0, 200]
    plt.figure()
    for i in range(15):
            plt.subplot(3,5,i+1)  # Subplot with 3*5 windows
            plt.imshow(img[:,:,index[i]],"gray")
            plt.axis('off')
            if ground_true[index[i]] == predict[index[i]]:
                    color="blue"
            else:
                    color = "red"

            plt.title(str(int(predict[index[i]])) +
                " (" + str(index[i]) + ")",color=color)

    plt.show()

if __name__ == '__main__':
    """----Initialize----"""
    while(1):
        NUMBER1 = int(input("Input NUMBER1 (0~9): "))
        if NUMBER1 not in range(10):
            print("NUMBER1 not in [0,10]")
        else:
            break

    while(1):
        NUMBER2 = int(input("Input NUMBER2 (0~9): "))
        if NUMBER2 not in range(10):
            print("NUMBER2 not in [0,10]")
        elif NUMBER1 ==  NUMBER2:
            print("NUMBER1 and NUMBER2 can not be the same.")
        else:
            break

    FILTER1 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
    FILTER2 = FILTER1.T
    """----Load The Data----"""
    image = np.zeros((28,28,200))
    image_resize = np.zeros((8,8,200))
    for i in range(100):
        image[:,:,i] = imread("train1000/"+
            str(100*NUMBER1+i+1)+".png",
            IMREAD_GRAYSCALE)
        image[:,:,i+100] = imread("train1000/"+
            str(100*NUMBER2+i+1)+".png",
            IMREAD_GRAYSCALE)
        image_resize[:,:,i] = resize(image[:,:,i], (8,8))
        image_resize[:,:,i+100] = resize(image[:,:,i+100], (8,8))
    """----Linear Regression----"""
    feature = feature_extraction(image_resize,FILTER1, FILTER2)
    target = np.concatenate((np.zeros((100,1)),np.ones((100,1))))
    weight = linear_regression(feature,target)
    target_predict = prediction(feature, target,weight)
    """---Show the result-----"""
    result(image, target, target_predict)