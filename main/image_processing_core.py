# import the necessary packages
import numpy as np
import cv2
import imutils
from imutils import contours
from PIL import Image

ROI = "roi/"

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
    im_arr_bgr = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness = 0)
    
    #Save object to image file
    roi = im_arr_bgr[y:y+h, x:x+w]
    cv2.imwrite(f"{ROI}section1.jpg", roi)

    #Get bounding rectangle of contour with second maximum area, and mark it with blue rectangle
    x, y, w, h = cv2.boundingRect(smax_c)
    ims_arr_bgr = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness = 0)
    
    #Save object to image file
    roi = ims_arr_bgr[y:y+h, x:x+w]
    cv2.imwrite(f"{ROI}section2.jpg", roi)
    
    #Get bounding rectangle of contour with third maximum area, and mark it with blue rectangle
    x, y, w, h = cv2.boundingRect(tmax_c)
    imt_arr_bgr = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness = 0)
    
    #Save object to image file
    roi = imt_arr_bgr[y:y+h, x:x+w]
    cv2.imwrite(f"{ROI}id.jpg", roi)
    
    print(get_id())

def get_id():
    # Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(f'{ROI}id.jpg')
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours and filter using contour area filtering to remove noise
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detect checkboxes using shape approximation and aspect ratio filtering
    checkbox_contours = []
    checkbox_answer = []
        
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if len(approx) == 4 and (aspect_ratio >= 0.8 and aspect_ratio <= 1.2) and (w >= 22 and w <= 24):
            # cv2.putText(original, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 3)
            checkbox_contours.append(c)

    # print('Checkboxes:', len(checkbox_contours))
    
    idCnts = contours.sort_contours(checkbox_contours)[0]
    idList = []
    
    for (q, i) in enumerate(np.arange(0, len(idCnts), 10)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # check answer
        cnts = contours.sort_contours(idCnts[i:i + 10],"top-to-bottom")[0]
        
        check = None
        
    # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "box" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # box area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if total > 300:
                if(j+1 < 10):
                    idList.append(j+1)
                else :
                    idList.append(0)
    
    if(len(idList)!=13):
        return False
    
    _id = "".join(str(item) for item in idList)
    return _id
    
def get_section1():
    return True

def get_section2():
    return True