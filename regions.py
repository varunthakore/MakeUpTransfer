import cv2
import dlib
from imutils import face_utils
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

def regions(image):
    
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the landmark (x, y)-coordinates to a NumPy array
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
        
    overlay = image.copy()
  
    
    for(i, name) in enumerate(face_utils.FACIAL_LANDMARKS_68_IDXS.keys()):
        (j, k) = face_utils.FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        if name == "mouth":
            hull1 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull1], -1, colors[i], -1)
        elif name == "inner_mouth":
            hull2 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull2], -1, colors[i], -1)
        elif name == "right_eye":
            hull3 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull3], -1, colors[i], -1)
        elif name == "left_eye":
            hull4 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull4], -1, colors[i], -1)
        elif name == "left_eyebrow":                 
            hull5 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull4], -1, colors[i], -1)
        elif name == "right_eyebrow":
            hull6 = cv2.convexHull(pts)
            #cv2.drawContours(overlay, [hull4], -1, colors[i], -1)
   
    
    plt.imshow(overlay)
 
    # #if not isinstance(hull[:,0],Delaunay):
    hull1 = Delaunay(hull1[:,0])
    hull2 = Delaunay(hull2[:,0])
    hull3 = Delaunay(hull3[:,0])
    hull4 = Delaunay(hull4[:,0])
    hull5 = Delaunay(hull5[:,0])
    hull6 = Delaunay(hull6[:,0])
    
    return [hull1, hull2, hull3, hull4, hull5, hull6]


