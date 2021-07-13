import dlib
from imutils import face_utils
import cv2
import numpy as np


def containsPoint(rect, p):
     if (p[0] < rect[0] or p[0] >= rect[2]) or (p[1] < rect[1] or p[1] >= rect[3]):
         return False
     else:
         return True
    
    
def warp(src, tar):
    
    src_c = src.copy()
    tar_c = tar.copy()
    src_mesh = src.copy()
    tar_mesh = tar.copy()
    
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    tar_gray = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
    src_forehead = [(24,43),(28,30), (31,4), (50,0), (114,0), (135,4),(137,11), (140,30), (145,49)]
    
    #src_forehead = [(25,41),(36,30),(46,17),(61,5),(85,1),(108,7),(114,16),(122,32),(125,42)] #exa 1
    
    #src_forehead = [(22 , 42),(27 , 24),(34 , 5),(49 , 2),(73 , 1),(93 , 1),(107 , 7),(114 , 20),(121 , 40)]#exa2
    
    for i in src_forehead:
        cv2.circle(src_c, (i[0],i[1]), 1, (0, 255, 0), -1)
    
    
    
    src_landmarks.extend(src_forehead)
    
    # landmarks for target image
    for (_, r) in enumerate(tar_rect):
        points = face_utils.shape_to_np(predictor(tar_gray, r))
        
        for (i1,i2) in points:
            cv2.circle(tar_c, (i1, i2), 1, (0, 255, 0), -1)
            tar_landmarks.append((i1,i2))
     
    
    
    # append 9 forehead points from left to right manually
    #tar_forehead = [(18,73),(24,59), (42,48), (54,40), (64,40), (103,39),(115,48),(128,59), (134,73)] #img 1
    tar_forehead = [(20,40),(24,15), (31,4), (54,0), (101,0), (118,4),(122,11),(132,30), (134,49)] #img 2
    #tar_forehead = [(23,42),(25,25),(41,8),(61,2),(94,2),(113,9),(117,24),(118,31),(124,51)]#sub1
    #tar_forehead  =[(19 , 42),(22 , 21),(26 , 6),(39 , 0),(72 , 1),(96 , 1),(112 , 7),(120 , 24),(121 , 40)]#sub2
    
   
    for i in tar_forehead:
        cv2.circle(tar_c, (i[0],i[1]), 1, (0, 255, 0), -1)
    
    tar_landmarks.extend(tar_forehead)
    
    
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
    cv2.imwrite("src_mesh1.png", src_mesh)
    
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
    cv2.imwrite("tar_mesh1.png", tar_mesh)
    
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
    
        
        # creating mask in the destination image
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
        m,n,_ = src.shape    
        res = cv2.warpAffine(src,matrix,(n,m))
        warp_img = cv2.bitwise_or(warp_img,cv2.bitwise_and(mask,res))
    
    
    
    return warp_img
    
