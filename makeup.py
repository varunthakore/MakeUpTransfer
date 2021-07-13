import cv2
import numpy as np
from matplotlib import pyplot as plt
       
        

#read images
src = cv2.imread("src1.png",cv2.COLOR_BGR2RGB)
src = cv2.resize(src, (160,226))
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)


tar = cv2.imread("tar2.png")
tar = cv2.resize(tar, (160,226))

tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

src_c = src.copy()
tar_c = tar.copy()
src_mesh = src.copy()
tar_mesh = tar.copy()

f, axarr = plt.subplots(2,2) 
axarr[0,0].imshow(src)
axarr[0,1].imshow(tar)


src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
tar_gray = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

#get aligned image
from Warp_img import warp
warp_img = warp(src, tar)
axarr[1,0].imshow(warp_img)

cv2.imwrite("warp.png", warp_img)

#Layer Decomposition
exa_img = warp_img.copy() #Example Image E
sub_img = tar.copy() #Subject Image I

exa_img_lab = cv2.cvtColor(exa_img, cv2.COLOR_RGB2LAB)


sub_img_lab = cv2.cvtColor(sub_img, cv2.COLOR_RGB2LAB)

Is = cv2.bilateralFilter(sub_img_lab[:,:,0],3,50,50) #Face structure layer
Id = sub_img_lab[:,:,0] - Is #Skin detail layer

Es = cv2.bilateralFilter(exa_img_lab[:,:,0],3,50,50) #Face structure layer
Ed = exa_img_lab[:,:,0] - Es #Skin detail layer

#Skin Detail Transfer
Rd = 0 * Id + 1 * Ed
#print(Id)
cv2.imwrite("skin_detail.png", Rd)

#create mask
mask1 = np.where(Es>0, 255, 0)



#Color Transfer
from regions import regions

[mouth,inner_mouth,right_eye,left_eye, left_eyebrow, right_eyebrow]=regions(tar)
#example mouth point
[exa_mouth,exa_inner_mouth,exa_right_eye,exa_left_eye,exa_left_eyebrow, exa_right_eyebrow]=regions(exa_img)


Ic_alpha = sub_img_lab[:,:,1]
Ic_beta = sub_img_lab[:,:,2]

Ec_alpha = exa_img_lab[:,:,1]
Ec_beta = exa_img_lab[:,:,2]

Rc_alpha = sub_img_lab[:,:,1]
Rc_beta = sub_img_lab[:,:,2]
m,n = Ic_alpha.shape
for i in range(m):
       for j in range(n):
        if mask1[i,j] == 255:
            if  mouth.find_simplex([j,i])>=0 or right_eye.find_simplex([j,i])>=0 or left_eye.find_simplex([j,i])>=0 :
                Rc_alpha[i,j] = Ic_alpha[i,j]
                Rc_beta[i,j] = Ic_beta[i,j]
            else:
                Rc_alpha[i,j] = (1-0.8) * Ic_alpha[i,j] + 0.8 * Ec_alpha[i,j]
                Rc_beta[i,j] = (1-0.8) * Ic_beta[i,j] + 0.8 * Ec_beta[i,j]
    
         
           
# color transfer
final_img = np.zeros((m,n,3))
final_img[:,:,0] = Is
final_img[:,:,1] = Rc_alpha
final_img[:,:,2] = Rc_beta

write_img = final_img.astype(np.uint8)
write_img = cv2.cvtColor(write_img, cv2.COLOR_LAB2BGR)


cv2.imwrite("color-transfer.png", write_img)


# Highlight and shading transfer
import math
from scipy.optimize import minimize
del_Is = cv2.Laplacian(Is,cv2.CV_64F, ksize = 1) #gradient of tar
del_Es = cv2.Laplacian(Es,cv2.CV_64F, ksize = 1) #gradient of src

def beta(p):
    p = np.array(p)
    sigma_sq = 160/25
    
    def objective(q):
        q = np.array(q)
        return [1-math.exp((-(np.linalg.norm(q-p))**2)/(2*sigma_sq))]
   
    res = minimize(objective,x0=[50,100],bounds=[(0,160),(0,200)])
    return res.fun[0]

del_Rs = np.zeros((m,n))   
for i in range(m):
    for j in range(n):
        if beta([i,j]) * abs(del_Es[i,j]) > abs(del_Is[i,j]):
            del_Rs[i,j] = del_Es[i,j]
        else:
            del_Rs[i,j] = del_Is[i,j]



Rs = del_Rs + Is

for i in range(m):
    for j in range(n):
        if mask1[i,j] == 255:
            
            final_img[i,j,0] = Rs[i,j]+ Rd[i,j]  #add skin detail layer
            
            if (mouth.find_simplex([j,i])>=0 or right_eye.find_simplex([j,i])>=0 or left_eye.find_simplex([j,i])>=0 or right_eyebrow.find_simplex([j,i])>=0 or left_eyebrow.find_simplex([j,i])>=0)\
                or (exa_mouth.find_simplex([j,i])>=0 or exa_right_eye.find_simplex([j,i])>=0 or exa_left_eye.find_simplex([j,i])>=0 or exa_right_eyebrow.find_simplex([j,i])>=0 or exa_left_eyebrow.find_simplex([j,i])>=0):
                
                final_img[i,j,0] = Is[i,j]
                final_img[i,j,1] = Ic_alpha[i,j]
                final_img[i,j,2] = Ic_beta[i,j]
        else:
            
                final_img[i,j,0] = Is[i,j]
                final_img[i,j,1] = Ic_alpha[i,j]
                final_img[i,j,2] = Ic_beta[i,j]
                
# remove error on the boundary of mask   
edges = np.uint8(mask1)
edges = cv2.Canny(edges,100,200)
edges = cv2.dilate(edges,(5,5),iterations = 5)
edges = cv2.dilate(edges,(5,5),iterations = 5)
for i in range(m):
    for j in range(n):
        if edges[i,j] == 255:
            final_img[i,j,0] = Is[i,j]
            final_img[i,j,1] = Ic_alpha[i,j]
            final_img[i,j,2] = Ic_beta[i,j]
            
            final_img[i,j-1,0] = Is[i,j]
            final_img[i,j-1,1] = Ic_alpha[i,j]
            final_img[i,j-1,2] = Ic_beta[i,j]
            
            final_img[i,j+1,0] = Is[i,j]
            final_img[i,j+1,1] = Ic_alpha[i,j]
            final_img[i,j+1,2] = Ic_beta[i,j]
               

final_img = final_img.astype(np.uint8)
final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)


write_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

cv2.imwrite("result_without_lips.png", write_img)



#Lip Makeup

exa_eq = cv2.equalizeHist(exa_img_lab[:,:,0])
sub_eq = cv2.equalizeHist(sub_img_lab[:,:,0])

#create list for lips point
sub_lips = []
exa_lips = []

#subject lips point
for i in range(m):
    for j in range(n):
        if mouth.find_simplex((j,i))>=0:
            sub_lips.append((i,j))




for i in range(m):
    for j in range(n):
        if exa_mouth.find_simplex((j,i))>=0:
            exa_lips.append((i,j))


          
def gaussian(x):
    return math.exp(-(float(x)**2)/2)

def lipMakeup(p):
    p = np.array(p, dtype='int')
    maxi = -1
    maxQ = (0,0)
    for i in exa_lips:
            q = i
            q = np.array(q, dtype='int')
            val = float(gaussian(np.linalg.norm(q-p)))
            val = val * gaussian((float((exa_eq[q[0],q[1]]) - float(sub_eq[p[0],p[1]])))/255)
            
            if val > maxi:
                maxi = val
                maxQ = q
    return maxQ


final_img_lab = cv2.cvtColor(final_img, cv2.COLOR_RGB2LAB)  

for k in sub_lips:
    (i,j) = k
    
    xcord,ycord = lipMakeup([i,j])
    xcord = int(xcord)
    ycord = int(ycord)

    #print(xcord, ycord)
    final_img_lab[i,j,0] = exa_img_lab[xcord,ycord,0]
    final_img_lab[i,j,1] = exa_img_lab[xcord,ycord,1]
    final_img_lab[i,j,2] = exa_img_lab[xcord,ycord,2]
mask1
 
           
final_img = cv2.cvtColor(final_img_lab, cv2.COLOR_LAB2BGR)  
cv2.imwrite("final_with_lips.png", final_img)
final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  
axarr[1,1].imshow(final_img) 
