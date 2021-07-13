import cv2
import numpy as np

def xdog(img, size=(0,0), sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
    img1 = cv2.GaussianBlur(img,size,sigma)
    img2 = cv2.GaussianBlur(img,size,sigma*k)
    img_filt = (img1-gamma*img2)/255
    for i in range(0,img_filt.shape[0]):
        for j in range(0,img_filt.shape[1]):
            if(img_filt[i,j] < epsilon):
                img_filt[i,j] = 1*255
            else:
                img_filt[i,j] = 255*(1 + np.tanh(phi*(img_filt[i,j])))
                
    return img_filt

img = cv2.imread('tar1.png',0)
img = cv2.resize(img,(160,200))
img_xdog = np.uint8(xdog(img,size=(0,0),sigma=0.4,k=1.6, gamma=0.5,epsilon=-0.5,phi=10))
cv2.imwrite('tar_xdog2.png',img_xdog)

