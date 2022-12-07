import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# Red-Green color blindness
img = cv.imread('images/test.png').astype(np.float32)/255.0
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
lab[:,:,1] *=0
img = (cv.cvtColor(lab, cv.COLOR_LAB2BGR))*255
img = img.astype(np.uint8)

fig = plt.figure()
plt.axis('off')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

if not os.path.exists('images'):
    os.mkdir('images')
cv.imwrite('images/ex2.2.1.jpg', img)

# Blue-Yellow color blindness
img = cv.imread('images/test.png').astype(np.float32)/255.0
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
lab[:,:,2] *=0
img = (cv.cvtColor(lab, cv.COLOR_LAB2BGR))*255
img = img.astype(np.uint8)

fig = plt.figure()
plt.axis('off')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

if not os.path.exists('images'):
    os.mkdir('images')
cv.imwrite('images/ex2.2.2.jpg', img)

# Glaucoma
img = cv.imread('images/test.png').astype(np.float32)/255.0

# Gaussian kernel
gh ,hr = divmod(np.size(img,0),2)
gw, wr = divmod(np.size(img,1),2)
x, y = np.mgrid[-gh:gh+hr, -gw:gw+wr]
sigma = 14**2
gaussian_kernel= np.exp(-(x**2+y**2)/(sigma**2))

#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
blur = np.zeros(img.shape)
fig = plt.figure()
plt.axis('off')
plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()

blur[:,:,0] = (img[:,:,0] * gaussian_kernel)
blur[:,:,1] = (img[:,:,1] * gaussian_kernel)
blur[:,:,2] = (img[:,:,2] * gaussian_kernel)
blur = (blur*255).astype(np.uint8)

fig = plt.figure()
plt.axis('off')
plt.imshow(cv.cvtColor(blur, cv.COLOR_BGR2RGB))
plt.show()

if not os.path.exists('images'):
    os.mkdir('images')
cv.imwrite('images/ex2.2.3.jpg', blur)
