# HW3

Using ``python`` to coding

Python Library:

* os
* numpy
* cv2
* matplotlib
* random
* datetime

```cmd
python HW3.py
```

## [HW3 - Image Recognition with Simplify CNN](HW3.ipynb)

[Dataset](https://mailntustedutw-my.sharepoint.com/:u:/g/personal/m11107309_ms_ntust_edu_tw/EfS2C1MOel5LpJ5J_ZUmngIBVGiOgaJuz0m4zxXDFwkSGw?e=IC6BPw): 1000 images(0~9)

Recognition with **linear regression**

## Method

![image](images/method.png)

### Load the data

Intput two number a & b (0~9)

![image](images/data.png)

### Create filter

Create two filters

$filter1=\begin{vmatrix}
-1  & -1 & 1   \\
-1  & 0  & 1   \\
-1  & 1  & 1   \\
\end{vmatrix}$

$filter2 = (filter1)^T$

### Feature extraction

```python
for i in range(200):

    # Leyer 1: Convolve2D
    conv1 = convolve2D(imgs[:,:,i],filter1)
    conv2 = convolve2D(imgs[:,:,i],filter2)
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
```

### Linear regression

$X(200\times9)$ : Feature

$Y(200\times1)$ : Target

$A(9\times1)$ : Optimal Coefficient

Let $Y=XA$

$A=(X^TX)^{-1}X^TY$

## Result

for **a = 0** and **b = 1**

### Confuse matrix

$\begin{vmatrix}
89  & 5  \\
11  & 95  \\
\end{vmatrix}$

### Prediction

![image](images/result.png)
