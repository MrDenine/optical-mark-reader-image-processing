# import the necessary packages
import numpy as np
import cv2
import imutils
from imutils import contours

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def cropPerspective(filename : str):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours and sort for largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    
    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            displayCnt = approx
            break

    # Obtain birds' eye view of image
    warped = four_point_transform(image,  np.array(displayCnt).reshape(4,2))
    
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("warped", warped)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    return warped

def find_squares(filename : str):
    # Read input image
    img = cv2.imread(filename)

    # Draw thick rectangle around the image - making sure there is not black contour around the image
    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), thickness = 5)

    # Convert from BGR to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold on gray image - use automatic threshold algorithm (use THRESH_OTSU) and invert polarity.
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours
    cnts, heir = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    max_a = 0  # Maximum area
    smax_a = 0 # Second maximum area
    tmax_a = 0 # Third maximum area
    fmax_a = 0 # Fourth maximum area

    max_c = []  # Contour with maximum area
    smax_c = [] # Contour with second maximum area (maximum excluding max_c)
    tmax_c = [] # Contour with third maximum area (maximum excluding max_c)
    fmax_c = [] # Contour with fourth maximum area (maximum excluding max_c)


    # Iterate contours
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_a:    # If area is grater than maximum, second max = max, and max = area
            smax_a = max_a
            smax_c = max_c  # Second max contour gets maximum contour
            max_a = area
            max_c = c       # Maximum contour gets c
        elif area > smax_a: # If area is grater than second maximum, replace second maximum
            smax_a = area
            smax_c = c
        elif area > tmax_a: # If area is grater than third maximum, replace third maximum
            tmax_a = area
            tmax_c = c
        elif area > fmax_a: # If area is grater than fourth maximum, replace fourth maximum
            fmax_a = area
            fmax_c = c

    #Get bounding rectangle of contour with maximum area, and mark it with green rectangle
    x, y, w, h = cv2.boundingRect(max_c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)

    #Get bounding rectangle of contour with second maximum area, and mark it with blue rectangle
    x, y, w, h = cv2.boundingRect(smax_c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness = 2)
    
    #Get bounding rectangle of contour with third maximum area, and mark it with blue rectangle
    x, y, w, h = cv2.boundingRect(tmax_c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness = 2)
    
    #Get bounding rectangle of contour with fourth maximum area, and mark it with blue rectangle
    x, y, w, h = cv2.boundingRect(fmax_c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness = 2)

    # Show result (for testing).
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_answer(filename : str, Answer_key):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours and sort for largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    
    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            displayCnt = approx
            break

    # Obtain birds' eye view of image
    warped = four_point_transform(gray,  np.array(displayCnt).reshape(4,2))
    thresh= cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
    cnts= cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts= imutils.grab_contours(cnts)
    questionCnts=[]
    for c in cnts:
        (x, y, w, h)= cv2.boundingRect(c)
        ar= w/ float(h)
        print(w,h,ar)
        if w>=20 and h>=20 and ar >=0.9 or ar<= 1.1:
            questionCnts.append(c)
            
    #sorting the contours from top to botton
    questionCnts= contours.sort_contours(questionCnts, method="top-to-botton")[0]
    correct= 0 

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts= contours.sort_contours(questionCnts[i: i+5]) [0]
        bubbled= None

        for (j,c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask= cv2.bitwise_and(thresh, thresh, mask=mask)
            total= cv2.countNonZero(mask)

            if bubbled is None or total >bubbled[0]:
                bubbled= (total, j)
            color= (0, 0, 255)
            k= Answer_key[q]    
            if k == bubbled[1]:
                correct= correct+1

    print(correct)  
    
        