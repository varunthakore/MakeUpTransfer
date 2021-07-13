import cv2
import dlib
import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt

def containsPoint(rect, p):
     if (p[0] < rect[0] or p[0] >= rect[2]) or (p[1] < rect[1] or p[1] >= rect[3]):
         return False
     else:
         return True
    
        
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#read images
src = cv2.imread("src1.png",cv2.COLOR_BGR2RGB)
src = cv2.resize(src, (160, 200))
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
tar = cv2.imread("tar_xdog.png")
tar = cv2.resize(tar, (160, 200))
tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

src_c = src.copy()
tar_c = tar.copy()
src_mesh = src.copy()
tar_mesh = tar.copy()

f, axarr = plt.subplots(2,3) 
axarr[0,0].imshow(src)
axarr[0,1].imshow(tar)


src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
tar_gray = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

#bounding rectangle
src_rect = detector(src_gray, 0)
tar_rect = detector(tar_gray, 0)

# find control points 

src_landmarks = []
tar_landmarks = []
#face landmark for source image
for (_, r) in enumerate(src_rect):
    points = face_utils.shape_to_np(predictor(src_gray, r))
    
    for (i1,i2) in points:
        cv2.circle(src_c, (i1, i2), 1, (0, 255, 0), -1)
        src_landmarks.append((i1,i2))
        

#append 9 forehead points points manually
src_forehead = [(24,43),(28,30), (31,4), (50,0), (114,0), (135,4),(137,11),
                        (140,30), (145,49)]

for i in src_forehead:
    cv2.circle(src_c, (i[0],i[1]), 1, (0, 255, 0), -1)



src_landmarks.extend(src_forehead)
axarr[0,2].imshow(src_c)

# landmarks for target image
for (_, r) in enumerate(tar_rect):
    points = face_utils.shape_to_np(predictor(tar_gray, r))
    
    for (i1,i2) in points:
        cv2.circle(tar_c, (i1, i2), 1, (0, 255, 0), -1)
        tar_landmarks.append((i1,i2))
 


# append 9 forehead points from left to right manually
tar_forehead = [(18,73),(24,59), (42,48), (54,40), (64,40), (103,39),(115,48),
                        (128,59), (134,73)]
#tar_forehead = [(20,40),(24,15), (31,4), (54,0), (101,0), (118,4),(122,11),(132,30), (134,49)]

for i in tar_forehead:
    cv2.circle(tar_c, (i[0],i[1]), 1, (0, 255, 0), -1)

tar_landmarks.extend(tar_forehead)

axarr[0,2].imshow(tar_c)

# # Show the image
# plt.imshow(tar)

#triangle mesh for src image
bounding_rect = (0, 0, src_gray.shape[1], src_gray.shape[0])
subdiv = cv2.Subdiv2D(bounding_rect)
for point in src_landmarks:
    if containsPoint(bounding_rect, point):
        subdiv.insert((point[0],point[1]))

src_triangles = subdiv.getTriangleList()
src_triangles = np.array(src_triangles, dtype=np.int32)


for t in src_triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    cv2.line(src_mesh, pt1, pt2, (192,192,192), 1)
    cv2.line(src_mesh, pt2, pt3, (192,192,192), 1)
    cv2.line(src_mesh, pt1, pt3, (192,192,192), 1)
    
src_mesh = cv2.cvtColor(src_mesh, cv2.COLOR_RGB2BGR)

#triangle mesh for tar image
bounding_rect = (0, 0, tar_gray.shape[1], tar_gray.shape[0])
subdiv = cv2.Subdiv2D(bounding_rect)
for point in tar_landmarks:
    if containsPoint(bounding_rect, point):
        subdiv.insert((point[0],point[1]))

tar_triangles = subdiv.getTriangleList()
tar_triangles = np.array(tar_triangles, dtype=np.int32)


for t in tar_triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    cv2.line(tar_mesh, pt1, pt2, (192,192,192), 1)
    cv2.line(tar_mesh, pt2, pt3, (192,192,192), 1)
    cv2.line(tar_mesh, pt1, pt3, (192,192,192), 1)
# Show the image
tar_mesh = cv2.cvtColor(tar_mesh, cv2.COLOR_RGB2BGR)


#warping src image to destination image
warp_img = np.zeros(src.shape, dtype=np.uint8)

for t in tar_triangles:
    tar_point1 = (t[0], t[1])
    tar_point2 = (t[2], t[3])
    tar_point3 = (t[4], t[5])
    
    src_point1 = (0,0)
    src_point2 = (0,0)
    src_point3 = (0,0)
    #find corresponding point in src image for every point in tar img
    for i in range(len(src_landmarks)):
        if tar_point1 == (tar_landmarks[i]):
            src_point1 = src_landmarks[i]
            
        if tar_point2 == (tar_landmarks[i]):
            src_point2 = src_landmarks[i]
            
        if tar_point3 == (tar_landmarks[i]):
            src_point3 = src_landmarks[i]

    
    # creating mask in destination image
    mask = np.zeros(src.shape, dtype=np.uint8)
    roi_corners = np.array([[tar_point1[0],tar_point1[1]],[tar_point2[0],tar_point2[1]]
                        ,[tar_point3[0],tar_point3[1]]], dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], (255,255,255))
    
    #warping src img to tar img
    pts1 = np.float32([[src_point1[0],src_point1[1]],[src_point2[0],src_point2[1]]
                        ,[src_point3[0],src_point3[1]]])
    pts2 = np.float32([[tar_point1[0],tar_point1[1]],[tar_point2[0],tar_point2[1]]
                        ,[tar_point3[0],tar_point3[1]]])
    
    matrix = cv2.getAffineTransform(pts1,pts2)
    rows,cols,ch = src.shape    
    res = cv2.warpAffine(src,matrix,(cols,rows))
    warp_img = cv2.bitwise_or(warp_img,cv2.bitwise_and(mask,res))

axarr[1,2].imshow(warp_img)

# #plt.imshow(warp_img)


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

print(Rd.shape)

#create mask
mask1 = np.where(Es>0, 255, 0)

#Color Transfer
from regions import regions
[mouth,inner_mouth,right_eye,left_eye, left_eyebrow, right_eyebrow]=regions(tar)

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
    
        


#Lip Makeup
import math
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

#example lips point
[exa_mouth,a,b,c,d,e]=regions(exa_img)


for i in range(m):
    for j in range(n):
        if exa_mouth.find_simplex((j,i))>=0:
            exa_lips.append((i,j))


          
def gaussian(x):
    return (1/(math.sqrt(2*math.pi)))*math.exp(-(float(x)**2)/2)

def lipMakeup(p):
    p = np.array(p, dtype='int')
    maxi = -1
    maxQ = (0,0)
    for i in exa_lips:
            q = i
            q = np.array(q, dtype='int')
            val = float(gaussian(np.linalg.norm(q-p)))
            val = val * gaussian((float(exa_eq[q[0],q[1]]) - float(sub_eq[p[0],p[1]]))/255)
            
            if val > maxi:
                maxi = val
                maxQ = q
    return maxQ

final_img = np.zeros((m,n,3))
final_img = np.zeros((m,n,3))
final_img[:,:,0] = Is
final_img[:,:,1] = Rc_alpha
final_img[:,:,2] = Rc_beta
final_img = final_img.astype(np.uint8)
final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)
axarr[1,1].imshow(final_img) 
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

           
final_img = cv2.cvtColor(final_img_lab, cv2.COLOR_LAB2BGR)  
axarr[1,2].imshow(final_img)   
cv2.imwrite("final_with_lips_tar.png", final_img)


